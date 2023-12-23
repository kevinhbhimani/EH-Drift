HDF5 file structure
-------------------
/ (Root Group)
│
├── /waveform (Dataset)
│   └── [Array of waveform data, one-dimensional]
│
├── /energy (Dataset)
│   └── [Single value, representing the energy]
│
├── /r (Dataset)
│   └── [Single value, representing the radius of event]
│
├── /z (Dataset)
│   └── [Single value, representing the height of the event]
│
├── /grid (Dataset)
│   └── [Single value, representing the grid size]
│
└── /detector_name (Dataset)
    └── [String, representing the name of the detector]

Attributes
----------
- surface_charge (Attribute)
  └── [Single value, representing the surface charge in 1e10 e/cm2]
  
- self_repulsion (Attribute)
  └── [Single value, indicating whether self-repulsion is considered]
