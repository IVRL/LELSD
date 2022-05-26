# CLIPLSD: Project for EPFL Course Computational Photography: CS 413

Abstract: \*\*

### Streamlit apps

A streamlit app is provided to view the trained directions depending on the chosen

```
streamlit run 2D_visualization.py
```

Using this app you can combine two directions found by the model for editing the same semantic and create a palette of
edit like figure bellow. Based on this palette you can choose the edit strengths that will yield your desired edit. Note
that you need to have a LELSD trained model with _num_latent_dirs=2_ to use this app.

## Acknowledgment

This project builds on top of [**Optimizing Latent Space Directions For GAN-based Local Image Editing**](https://github.com/IVRL/LELSD)
