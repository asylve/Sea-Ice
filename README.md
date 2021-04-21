# Hudson Bay Sea Ice Segmentation - Project Summary

-  Collected 3392 satelite images of Hudson Bay sea ice in the Candian Arctic from 2016-1-1 to 2017-7-31
-  Generated sea ice concentrations masks for each image using Canadian Regional Ice Chart shapefiles
-  Used image/mask pairs to train a Convolutional Neural Network (U-Net) to segment satellite images based on seven different classes (7 levels of ice concentration and land)
    -  Model Accuracy:
    -  Model Mean Class IoU (intersection over union) score: 
 
# Code/Resources

**Python Version:** 3.7.10  
**Libraries Used:** eolearn, sentinelhub, numpy, pandas, matplotlib, geopandas, sklearn, tensorflow, keras

**EO-Learn Satellite Image Collection and Cleaning:** https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html 

**Ice Chart Masks:** Canadian Ice Service, . 2009. Canadian Ice Service Arctic Regional Sea Ice Charts in SIGRID-3 Format, Version 1. Subset: Hudson Bay Regional Ice Charts. Boulder, Colorado USA. NSIDC: National Snow and Ice Data Center. doi: https://doi.org/10.7265/N51V5BW9. Date Accessed: March 27, 2021.

# 1. Data Collection

There are two main data sources for this project: Sentinel-2 satellite images and Canadian Regional Ice Charts. These were used to generate images and masks, respectively.

## 1.1 Sentinel-2 

The Sentinel-2 mission is made up of a pair of optical satellites that image the globe roughly every 5 days. They capture images in 12 optical bands including the visible spectrum.

## 1.2 Canadian Regional Ice Charts

Canadian Regional Ice Charts show geospacial sea ice concentrations for ship safety and environmental monitoring. They are produced weekly on Mondays by the Canadian Ice Service for five large regions:

- Hudson Bay
- Western Arctic
- Eastern Arctic
- Eastern Coast
- Great Lakes

A sample ice chart for Hudson Bay on April 12, 2021 is shown below. Each region on the chart has a corresponding set of codes giving information on, among other things, the concentration of sea ice. The chart below right shows the codes corresponding to ice concentration ([source])(https://library.wmo.int/doc_num.php?explnum_id=9270). All charts are archived and available as shapefiles from the National Snow and Ice Data Centre dating back to 2006.

<p float="left">
  <img src="/Images/Ice_Chart_ex.gif" width="400" /> 
  <img src="/Images/Chart Codes.PNG" width="400" />  
</p>


## 1.3 Data Collection Workflow

Data was collected using the EO-Learn python library, which provides a framework for slicing large geographical areas into smaller, more manageable tiles called EOPatches. This allows for creating a data collection pipeline where satellite images are aquired through the Sentinelhub API. The workflow can also include filtering steps to avoid cloudy images as well as custom steps to add additional features such as image masks. The Data collection workflow loops over each EOPatch and consists of:

- **add_data:** Collect all available satellite images for the EOPatch in false color (bands B03, B04, and B08)
- **remove_dates:** Discard images that were taken more than 36 hours away from an available ice chart
- **add_valid_mask:** Collect a mask for each image that says which pixels are valid data
- **add_coverage:** Collect a mask for each image that says which pixels are blocked by clouds
- **remove_cloudy_scenes:** Remove images where the sum of cloudy and non-valid pixels is greater than 5%
- **time_raster:** Custom task to locate the ice chart temporally closest to the image, locate the area of the chart associated with the image, and rasterize into an ice concentration mask for the image
- **save_im:** Save each image and mask 

<p float="left">
  <img src="/Images/Region-Grid.png" width="800" /> 
  <img src="/Images/image-mask.png" width="800" />  
</p>

# 2. Data Cleaning


# 3. Exploratory Data Analysis (EDA)



# 4 Model Building


