from google.cloud import bigquery
import json
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

class GenericBigQueryClient:
    def __init__(self):
        """
        Initialize a generic BigQuery client
        
        """
        self.client = bigquery.Client()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_query_template(self, query_file: str) -> str:
        """
        Load SQL query template from file
        
        Args:
            query_file: Path to the query file (relative to script directory or absolute)
            
        Returns:
            SQL query template as string
        """
        try:
            # Handle both relative and absolute paths
            if not os.path.isabs(query_file):
                # Look in script directory first, then parent directory
                script_dir = os.path.dirname(__file__)
                query_path = os.path.join(script_dir, query_file)
                
                if not os.path.exists(query_path):
                    query_path = os.path.join(os.path.dirname(script_dir), query_file)
            else:
                query_path = query_file
            
            with open(query_path, 'r') as f:
                query_template = f.read()
            
            self.logger.info(f"Loaded query template from: {query_path}")
            return query_template
            
        except FileNotFoundError:
            self.logger.error(f"Query file not found: {query_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading query template: {str(e)}")
            raise
    
    def replace_date_placeholders(self, query: str, start_date: str, end_date: str) -> str:
        """
        Replace date placeholders in query
        
        Args:
            query: SQL query string
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Query with date placeholders replaced
        """
        # Common date placeholder patterns
        replacements = {
            "@start_date": f"DATE '{start_date}'",
            "@end_date": f"DATE '{end_date}'",
            "@startDate": f"DATE '{start_date}'",
            "@endDate": f"DATE '{end_date}'",
            "{{start_date}}": f"DATE '{start_date}'",
            "{{end_date}}": f"DATE '{end_date}'",
            # Legacy support for existing patterns
            "DATE '2025-01-01'": f"DATE '{start_date}'",
            "DATE '2025-03-31'": f"DATE '{end_date}'"
        }
        
        for placeholder, replacement in replacements.items():
            query = query.replace(placeholder, replacement)
        
        return query
    
    def load_group_list_from_json(self, json_file: str, json_path: List[str]) -> str:
        """
        Load group members from JSON file and format for SQL
        
        Args:
            json_file: Path to JSON file
            json_path: List of keys to navigate through JSON structure
            
        Returns:
            Formatted string for SQL IN clause
        """
        try:
            # Handle relative paths
            if not os.path.isabs(json_file):
                script_dir = os.path.dirname(__file__)
                json_file = os.path.join(script_dir, json_file)
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Navigate through JSON structure using path
            current_data = data
            for key in json_path:
                current_data = current_data.get(key, {})
            
            # Extract list (assuming it's a list at the end of the path)
            if isinstance(current_data, list):
                members = current_data
            else:
                self.logger.warning(f"Expected list at JSON path {json_path}, got {type(current_data)}")
                return "''"
            
            # Format for SQL
            formatted_list = ", ".join([f"'{member}'" for member in members])
            
            self.logger.info(f"Loaded {len(members)} group members from {json_file}")
            return formatted_list
            
        except Exception as e:
            self.logger.error(f"Error loading group list from JSON: {str(e)}")
            return "''"
    
    def replace_group_placeholders(self, query: str, group_list: str) -> str:
        """
        Replace group list placeholders in query
        
        Args:
            query: SQL query string
            group_list: Comma-separated, quoted list of group members
            
        Returns:
            Query with group placeholders replaced
        """
        replacements = {
            "@group_list": group_list,
            "@groupList": group_list,
            "{{group_list}}": group_list,
            "@group": group_list
        }
        
        for placeholder, replacement in replacements.items():
            query = query.replace(placeholder, replacement)
        
        return query
    
    def replace_custom_placeholders(self, query: str, parameters: Dict[str, Any]) -> str:
        """
        Replace custom placeholders in query
        
        Args:
            query: SQL query string
            parameters: Dictionary of parameter name -> value mappings
            
        Returns:
            Query with custom placeholders replaced
        """
        for param_name, param_value in parameters.items():
            # Support multiple placeholder formats
            placeholders = [
                f"@{param_name}",
                f"{{{{{param_name}}}}}",
                f"${param_name}"
            ]
            
            # Convert value to appropriate SQL format
            if isinstance(param_value, str):
                sql_value = f"'{param_value}'"
            elif isinstance(param_value, (int, float)):
                sql_value = str(param_value)
            elif isinstance(param_value, list):
                sql_value = ", ".join([f"'{item}'" if isinstance(item, str) else str(item) for item in param_value])
            else:
                sql_value = str(param_value)
            
            for placeholder in placeholders:
                query = query.replace(placeholder, sql_value)
        
        return query
    
    def execute_query(self, 
                     query_file: str,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     group_list: Optional[str] = None,
                     group_json_config: Optional[Dict[str, Any]] = None,
                     custom_parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Execute BigQuery with flexible parameter replacement
        
        Args:
            query_file: Name or path of the SQL query file
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            group_list: Pre-formatted group list string (optional)
            group_json_config: Dict with 'file' and 'path' keys for loading from JSON (optional)
            custom_parameters: Dictionary of custom parameters to replace (optional)
            
        Returns:
            List of query result dictionaries
        """
        try:
            # Load query template
            query = self.load_query_template(query_file)
            
            # Replace date parameters if provided
            if start_date and end_date:
                query = self.replace_date_placeholders(query, start_date, end_date)
                self.logger.info(f"Applied date range: {start_date} to {end_date}")
            
            # Handle group list
            if group_list:
                # Use provided group list
                query = self.replace_group_placeholders(query, group_list)
                self.logger.info("Applied provided group list")
            elif group_json_config:
                # Load group list from JSON
                loaded_group_list = self.load_group_list_from_json(
                    group_json_config['file'],
                    group_json_config['path']
                )
                query = self.replace_group_placeholders(query, loaded_group_list)
                self.logger.info("Applied group list from JSON")
            
            # Replace custom parameters if provided
            if custom_parameters:
                query = self.replace_custom_placeholders(query, custom_parameters)
                self.logger.info(f"Applied custom parameters: {list(custom_parameters.keys())}")
            
            self.logger.info(f"Executing query from {query_file}")
            
            # Execute query
            query_job = self.client.query(query)
            results = query_job.result()
            
            # Convert to list of dictionaries
            data = [dict(row) for row in results]
            self.logger.info(f"Retrieved {len(data)} records from BigQuery")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error executing BigQuery: {str(e)}")
            raise