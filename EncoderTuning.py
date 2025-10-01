from pythonnet import load
load("coreclr")

# Import System for Type.GetType
import System
from System.Collections.Generic import List
from System import String

import clr

import automation1 as a1
import sys
import contextlib
import os
import time
import numpy as np
#import serial.tools.list_ports
import tkinter as tk
from tkinter import messagebox, filedialog
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
import zipfile
import xml.etree.ElementTree as ET
import shutil


class EncoderTuning():
    """
    Class to handle encoder tuning for a controller.

    This class manages the tuning of encoder values for a given axis based
    on encoder sine and cosine values gathered during a move.

    It also provides methods to interact with the controller for commanding motion, 
    gathering data, and modifying parameters.
    """
    def __init__(self, controller, axes):
        """
        Initializes the EncoderTuning class with a controller and axes.
        
        Args:
        controller (a1.Controller): A1 controller object.
        axes (list): List of axes being tested.
        """
        self.controller = controller
        self.axes = axes
    

    def initialize_dll(self):
        """
        Set up and initialize DLL paths

        Args:
        None

        Returns:
        None
        """
        # Add Aerotech DLL directory to PATH
        self.AEROTECH_DLL_PATH = os.path.join(os.path.dirname(__file__), "extern", "Automation1")
        if not os.path.exists(self.AEROTECH_DLL_PATH):
            print(f"ERROR: Aerotech DLL path not found: {self.AEROTECH_DLL_PATH}")
            return
    
        # Add ConfigurationManager path
        self.CONFIG_MANAGER_PATH = os.path.join(os.path.dirname(__file__), "System.Configuration.ConfigurationManager.8.0.0", "lib", "netstandard2.0")
        if not os.path.exists(self.CONFIG_MANAGER_PATH):
            print(f"ERROR: ConfigurationManager not found at {self.CONFIG_MANAGER_PATH}")
            return
    
        os.environ["PATH"] = self.AEROTECH_DLL_PATH + ";" + os.environ["PATH"]
        os.add_dll_directory(self.AEROTECH_DLL_PATH)
    
    def load_dll(self):
        try:
            # Load ConfigurationManager
            print("\nLoading ConfigurationManager...")
            clr.AddReference(os.path.join(self.CONFIG_MANAGER_PATH, "System.Configuration.ConfigurationManager.dll"))
        
            # Then load the Aerotech DLLs
            print("Loading Aerotech assemblies...")
            clr.AddReference(os.path.join(self.AEROTECH_DLL_PATH, "Aerotech.Automation1.Applications.Core.dll"))
            print("Aerotech.Core loaded successfully")
            clr.AddReference(os.path.join(self.AEROTECH_DLL_PATH, "Aerotech.Automation1.Applications.Interfaces.dll"))
            print("Aerotech.Interfaces loaded successfully")
            clr.AddReference(os.path.join(self.AEROTECH_DLL_PATH, "Aerotech.Automation1.Applications.Shared.dll"))
            print("Aerotech.Shared loaded successfully")
            clr.AddReference(os.path.join(self.AEROTECH_DLL_PATH, "Aerotech.Automation1.DotNetInternal.dll"))
            print("Aerotech.DotNetInternal loaded successfully")
            clr.AddReference(os.path.join(self.AEROTECH_DLL_PATH, "Aerotech.Automation1.Applications.Wpf.dll"))
            print("Aerotech.Wpf loaded successfully")

            # Get the types using assembly-qualified names
            type_name1 = "Aerotech.Automation1.Applications.Shared.EllipseFit, Aerotech.Automation1.Applications.Shared"
            type_name2 = "Aerotech.Automation1.Applications.Shared.EllipseData, Aerotech.Automation1.Applications.Shared"
            type_name3 = "Aerotech.Automation1.Applications.Shared.EncoderGains, Aerotech.Automation1.Applications.Shared"
            type_name4 = "Aerotech.Automation1.Applications.Shared.EncoderTuningMotionConfiguration, Aerotech.Automation1.Applications.Shared"
            self.EllipseFit = System.Type.GetType(type_name1)
            self.EllipseData = System.Type.GetType(type_name2)
            self.EncoderGains = System.Type.GetType(type_name3)
            self.EncoderTuningMotionConfiguration = System.Type.GetType(type_name4)
        
            if self.EllipseFit is None or self.EllipseData is None or self.EncoderGains is None or self.EncoderTuningMotionConfiguration is None:
                print("\nFATAL: One or both types could not be found. They may not be public, or the namespace may be different.")
                return
        
            print("Successfully loaded required types")
        except Exception as e:
            print("\nERROR: An exception occurred:")
            print(str(e))
            print("\nException type:", type(e).__name__)
            if hasattr(e, 'InnerException') and e.InnerException:
                print("\nInner Exception:")
                print(str(e.InnerException))

    def calculate_angular_displacement(self, lines_per_rev, desired_cycles):
        """
        Calculates the angular displacement for a rotary encoder.

        Args:
        lines_per_rev (int): The number of lines or pulses per revolution for the encoder.
        desired_cycles (int or float): The number of encoder cycles you want to move.

        Returns:
        float: The total angular displacement in degrees.
        """
        if lines_per_rev <= 0:
            raise ValueError("Lines per revolution must be a positive number.")
    
        # In a standard encoder, one cycle corresponds to one line.
        cycles_per_rev = lines_per_rev
  
        # Calculate the degrees of rotation for a single cycle.
        degrees_per_cycle = 360 / cycles_per_rev
  
        # Calculate the total angular displacement for the desired number of cycles.
        total_displacement = degrees_per_cycle * desired_cycles
  
        return total_displacement

    def calculate_linear_displacement(self, resolution, desired_cycles, units="mm"):
        """
        Calculates the linear displacement for a linear encoder.

        Args:
        resolution (float): The distance moved per encoder cycle (e.g., 0.005 for 5µm).
                            Ensure this value is in the same unit as `units`.
        desired_cycles (int or float): The number of encoder cycles you want to move.
        units (str, optional): The measurement unit for the output. Defaults to "mm".

        Returns:
        str: A formatted string with the total linear displacement and its units.
        """
        if resolution <= 0:
            raise ValueError("Resolution must be a positive number.")

        # Calculate the total distance moved.
        total_distance = resolution * desired_cycles
  
        return total_distance

    def data_config(self, n: int, freq: a1.DataCollectionFrequency) -> a1.DataCollectionConfiguration:
        """
        Data configurations. These are how to configure data collection parameters

        Args:
        n (int): Number of points to collect (sample rate * time).
        freq (a1 object): Data collection frequency (limited to the available frequencies listed in the Studio dropdown).

        Returns:
        data_config: A1 data collection object used for data collection in A1
        """
        # Create a data collection configuration with sample count and frequency
        data_config = a1.DataCollectionConfiguration(n, freq)

        # Add items to collect data on the entire system
        data_config.system.add(a1.SystemDataSignal.DataCollectionSampleTime)

        for axis in self.axes:
                # Add items to collect data on the specified axis
                data_config.axis.add(a1.AxisDataSignal.EncoderCosine, axis)
                data_config.axis.add(a1.AxisDataSignal.EncoderSine, axis)
                data_config.axis.add(a1.AxisDataSignal.EncoderCosineRaw, axis)
                data_config.axis.add(a1.AxisDataSignal.EncoderSineRaw, axis)
        
        return data_config

    def generate_axis_specs(self):
        """
        Generates specifications for each axis including type, encoder type, resolution, and max velocity.
        Returns: 
        axis_specs (dict): A dictionary containing axis specifications.
        axes_to_tune (list): A list of axes that can be tuned.
        """
        axis_specs = {}
        axes_to_tune = []
        for axis in self.axes:
            axis_specs[axis] = {}
            # Determine if rotary or linear axes
            units_value = self.controller.runtime.parameters.axes[axis].units.unitsname.value
            if units_value == 'deg':
                axis_specs[axis]['Stage Type'] = 'rotary'
            else:
                axis_specs[axis]['Stage Type'] = 'linear'

            # Determine if sine or square wave
            wave_type = int(self.controller.runtime.parameters.axes[axis].feedback.primaryfeedbacktype.value)
            if (wave_type & 1 << 2):
                axis_specs[axis]['Encoder Type'] = 'sine'
                axes_to_tune.append(axis)
            if (wave_type & 1 << 3):
                axis_specs[axis]['Encoder Type'] = 'sine'
                axes_to_tune.append(axis)
            if (wave_type & 1 << 10):
                axis_specs[axis]['Encoder Type'] = 'sine'
                axes_to_tune.append(axis)
        
            # Get resolution for distance
            resolution = self.controller.runtime.parameters.axes[axis].feedback.primaryfeedbackresolution.value
            axis_specs[axis]['Resolution'] = resolution

            # Get max velocity
            max_velocity = self.controller.runtime.parameters.axes[axis].motion.maxspeedclamp.value
            axis_specs[axis]['Max Velocity'] = max_velocity

        return axis_specs, axes_to_tune

    def encoder_tuning(self):
        # Get axis specs and list of axes to tune
        axis_specs, axes_to_tune = self.generate_axis_specs()

        results = {}
        for axis in axes_to_tune:
            if axis_specs[axis]['Stage Type'] == 'rotary':
                distance = self.calculate_angular_displacement(axis_specs[axis]['Resolution'], 50)
            else:
                distance = self.calculate_linear_displacement(axis_specs[axis]['Resolution'], 50)
        
        speed = speed if axis_specs[axis]['Max Velocity'] > speed else axis_specs[axis]['Max Velocity']

        distance_to_analyze = distance * 0.5  # Use half the distance for analysis

        axis_data = {}
         # Load the DLL containing the EllipseFit class
        self.load_dll()

        for axis in self.axes:
            axis_data[axis] = {}
            tuning_call = self.EncoderTuningMotionConfiguration.GetMethod("CreateWithMotion")
            tuning = tuning_call.Invoke(None, [distance, speed, distance_to_analyze])

            axis_data[axis] = tuning

        return axis_data

    def fit_ellipse(self, signal_dict):
        """
        Use encoder data to fit an ellipse and get new encoder gains.

        Args:
            signal_dict (dict): Dictionary of encoder signals organized by axis and signal name.

        Returns:
            axis_ellipse_data (dict): Dictionary containing the ellipse fit result for each axis.
        """
        # Load the DLL containing the EllipseFit class
        self.load_dll()

        # Initialize the ellipse fit data structure
        axis_ellipse_data = {}

        # Iterate through each axis and fit the ellipse using sine and cosine data
        for axis, signals in signal_dict.items():
            # Check if both sine and cosine signals are present for the axis
            if 'Encoder Sine' in signals and 'Encoder Cosine' in signals:
                # Extract sine and cosine data
                sine_data = signals['Encoder Sine']
                cosine_data = signals['Encoder Cosine']
                # Create an instance of the EllipseFit class
                fit_call = self.EllipseFit.GetMethod("Fit")
                # Pass the sine and cosine data to the Fit method
                fit = fit_call.Invoke(None, [sine_data, cosine_data])
                # Populate the axis_ellipse_data dictionary with the fit result
                axis_ellipse_data[axis] = fit
                print(f"Ellipse fit complete for {axis}")

        return axis_ellipse_data

    def collect_data(self):
        """
        Move axis and collect encoder data that will be used to fit the ellipse.
        
        Returns:
        results (dict): A dictionary of results keyed by axis name.
        """
        # Get axis specs and list of axes to tune
        axis_specs, axes_to_tune = self.generate_axis_specs()

        results = {}
        for axis in axes_to_tune:
            if axis_specs[axis]['Stage Type'] == 'rotary':
                distance = self.calculate_angular_displacement(axis_specs[axis]['Resolution'], 50)
            else:
                distance = self.calculate_linear_displacement(axis_specs[axis]['Resolution'], 50)
        
            speed = speed if axis_specs[axis]['Max Velocity'] > speed else axis_specs[axis]['Max Velocity']
        
            time = distance / speed        
        
            # Setup Data Collection
            sample_rate = 1000
            n = int(sample_rate * time)
            freq = a1.DataCollectionFrequency.Frequency1kHz

            # Check if axis is enabled
            self.controller.runtime.commands.motion.enable([axis])

            # Collect data
            config = self.data_config(n, freq, axis=axis)
            self.controller.runtime.data_collection.start(a1.DataCollectionMode.Snapshot, config)
            self.controller.runtime.commands.motion.moveabsolute([axis], [distance], [speed])
            self.controller.runtime.commands.motion.waitformotiondone(axis)
            time.sleep(3)
            self.controller.runtime.data_collection.stop()
            results[axis] = self.controller.runtime.data_collection.get_results(config, n)

        return results

    def gather_results(self, results):
        """
        Gathers data for all axes tested and populates a dictionary organized by signal name and axis

        Args:
        results (dict): A dictionary of data collection results organized by axis name.

        Returns:
        signal_data_dict (dict): A dictionary of all encoder data signals organized by signal name and axis.
        """
        # Define the signals with separate titles
        signals = [
            (a1.AxisDataSignal.EncoderSine, 'Encoder Sine'),
            (a1.AxisDataSignal.EncoderCosine, 'Encoder Cosine'),
            (a1.AxisDataSignal.EncoderSineRaw, 'Encoder Sine Raw'),
            (a1.AxisDataSignal.EncoderCosineRaw, 'Encoder Cosine Raw')
        ]

        # Compile Results for each axis
        axis_data_dict = {}
        for axis, data in results.items():
            print(f"📈 Processing {axis} encoder data...")

            # Initialize the inner dictionary for this axis
            axis_data_dict[axis] = {}

            # Populate signal dictionary with data
            for signal_type, _ in signals:
                # Get data for this axis and signal using the correct method
                signal_data = data.axis.get(signal_type, axis).points
                signal_array = np.array(signal_data).tolist()

                # Store signal data for this axis and signal
                axis_data_dict[axis][signal_type.name] = signal_data

        return axis_data_dict

    def test(self):
        """
        Entry function to begin encoder tuning logic.

        :returns: None
        """
        print('Beginning Encoder Tuning Sequence')

        # Execute data collection
        #results = self.collect_data()

        # Compile and organize signals
        #signal_dict = self.gather_results(results)

        # Initialize DLLs
        self.initialize_dll()

        # Fit ellipse
        #ellipse_data = self.fit_ellipse(signal_dict)

        # Run test encoder tuning function
        results = self.encoder_tuning()
        print(f"Encoder tuning results: {results}")





