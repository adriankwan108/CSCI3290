Requirement:    
<pre>
Python 3.4+     
numpy   
opencv
</pre>

**Background:**     
<pre>
To combine a set of images into a large image, 
we could execute through registering, warping, resampling and blending.
In general, this technique is called Image Stitching.
There are two types of image stitching: direct / feature-based.

This programme focuses on feature-based and average blending.
*There are other kinds of blending, e.g. pyramid blending, alpha blending.
</pre>

**Usage:**    
<pre>
Combine two images.
</pre>

**Algorithm:**  
<pre>
Detect SIFT features. (Scale-Invariant Feature Transform)
Establish feature correspondence bewteen SIFT features in the two inputs using Brute-Force matcher.
Apply ratio test to select the set of robust matched points.
Estimate the best homography by employing RANSAC algorithm to eliminate the bad matches. (Random Sample Consensus)
    RANSAC:
    Select four feature pairs at random
    Compute homography H
    Compute inliers where  ||pi´, H pi|| < ε
    Keep largest set of inliers
    Re-compute least squares H estimate using al of the inliers
Wrap img_2 using the best homography to align with img_1 using inverse warping and bilinear resampling.
Stitch img_2 to img_1 and apply average blending, make two images into a single panorama image.

</pre>

**Command:**  
<pre>
python3 image_stitching.py --im1 /PATH/TO/INPUT/IMAGE1 --im2 /PATH/TO/INPUT/IMAGE2 --output /PATH/TO/OUTPUT/IMAGE
</pre>