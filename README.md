## HDF5 File Structure

This section describes the structure of the HDF5 file used in the project.

### Root Group

The root group `/` contains the following datasets:

- `/waveform`: Array of waveform data (one-dimensional, storing scaled signal values).
- `/energy`: Single value representing the energy.
- `/r`: Single value representing the radius of event.
- `/z`: Single value representing the height of event.
- `/grid`: Single value representing the grid size.
- `/detector_name`: String representing the name of the detector.

### Attributes

The following attributes are attached to the root group:

- `surface_charge`: Single value representing the surface charge.
- `self_repulsion`: Single value indicating whether self-repulsion is considered.
