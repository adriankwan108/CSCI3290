Requirement:    
<pre>
Python 3.4+     
numpy   
imageio     
</pre>

**Background:**     
<pre>
Practice the foundation of digital image processing, computer vision and image filering.
Image processing operations implemented with filtering include smoothing, sharpening, and denoising.
This program focuses on sharpening.
</pre>

**Usage:**    
<pre>
Use linear filering to sharpen a PNG format GRAYSCALE image.
</pre>

**Algorithm:**  
<pre>
Read input  
Generate the Gaussian kernel    
Convolve the input image    
Generate the sharpened image with smoothed image and original input     
    i.   crop input     
    ii.  detail map = crop_image - smoothed_image   
    iii. sharpened image = crop_image + detail map  
    iv.  return sharpened_image 
Output  
</pre>

**Command:**  
<pre>
python3 sharpening.py --input /PATH/TO/INPUT/IMAGE --kernel KERNEL_SIZE --sigma SIGMA --output /PATH/TO/OUTPUT/IMAGE

*sigma is for adjusting the gaussian kernel

i.e. python3 sharpening.py --input test_01.png --kernel 5 --sigma 1.5 --output output_01.png
</pre>
