
# Thermal Insulating Floater Oscillations under Rayleigh-Bénard Convection

This repository contains the code, bibliography, and experimental media for the study of a thermal insulating floater's periodic oscillations driven by Rayleigh-Bénard convection.

## Project Overview
The project involved designing an experimental setup from scratch to visualize fluid dynamics. We utilized shadowgraphy (ombroscopie) to capture the convective flow and applied computer vision techniques to extract the floater's kinematics and the fluid's velocity fields.

## Repository Structure

* `code/`: Python scripts computing spatial-temporal correlations on shadowgraphy sequences to analyze the fluid motion and extract the floater's oscillation period.
* `bibliography/`: Key reference papers and literature used to model the Rayleigh-Bénard convection mechanism.
* `media/`: 
  * Two raw experimental images of the physical setup.
  *  A time-lapse sequence of the floater's motion. 

## About the Media Processing
The `processed_oscillation.gif` was generated from a 600-frame sequence processed via ImageJ. A thresholding technique and a white background were applied to isolate and highlight the floater's kinematics. 
*Note on playback speed:* The sequence is encoded at 30 fps and further accelerated by a factor of 1.3x, resulting in an effective playback rate of 39 frames per second (a 39x acceleration factor assuming a standard 1 fps experimental capture).
