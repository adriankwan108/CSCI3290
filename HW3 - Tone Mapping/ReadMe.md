Requirement:    
<pre>
Python 3.4+     
numpy   
opencv   
</pre>

**Background:**     
<pre>
To map one set of colors to another to approximate the appearance of HDR,
in a medium that has a more limited dynamic range (LDR),
tone mapping is used.
In addition, it could address the problem of strong contrast reduction from the scene radiance to the displayable range 
while preserving the image details and color appearance.

There are two types of tone mapping operators:
1. global operators:
        Non-linear functions (log/exp/...) based on the luminance and other global variables of the image.
        => good for capturing overall preview
        
2. local operators:
        The parameters of the non-linear function change in EACH pixel, 
        according to features extracted from the surronding parameters.
        => can emphasize more local details

* Hence, local operators are far more expensive than global operators.
  For gaming, real time graphics, global tone mapping is used widely,
  for digital photography, local tone operators are used, 
        but local TMOs can give strange results if applied to video.

There are several kinds of tone mapping:

Global: CE tone mapping (CryEngine2, using expotential), Filmic tone mapping (Uncharted 2), 
Academy Color Encoding System (ACES tone mapping)(Rise of the Tomb Raider, UE4.8), etc.

Local: Reinhard, Durand

</pre>

**Usage:**    
<pre>
The program has both global (log)and local (Durand)tone mapping operators for transforming a HDR to LDR.
</pre>

**Algorithm:**          
*Overall:*
<pre>
1. Load HDR_Image
2. Apply tone mapping funciton to get tone mapped LDR image.
3. Apply gamma correction.
4. Convert LDR to 8bit.
</pre>

*Tone Mapping function:*
<pre>
1. Compute the Luminance L of each pixel in HDR.
2. Apple Tone mapping operator to L, compute the Display Luminance D for each pixel.
3. Map display luminance D on HDR to compose LDR
4. Return LDR
</pre>

*Tone mapping operator (Global):* Logarithmetic
<pre>
𝜏 = 𝛼(𝐿𝑚ax − 𝐿𝑚in )
𝛼 >= 0, 
this will tune the overall brightness of output, we assume 0.05

𝐷 = (log(𝐿 + 𝜏) − log (𝐿𝑚in + 𝜏) ) /
    (log(𝐿𝑚ax + 𝜏) − log (𝐿𝑚in + 𝜏))
𝐿𝑚ax, 𝐿𝑚in are the max and min luminance of the scene.
</pre>

*Tone mapping operator (Local):* Durand
<pre>
1. Compute the log intensity log10(𝐿)
2. Use bilateral filter, filtering the log intensity to get base layer: 𝐵aseLayer = bilateral_filter(log10(𝐿))
3. Decompose the detail layer: DetailLayer = log10(𝐿) - BaseLayer
4. Compute 𝛾 = log10(contrast) / 
                max(BaseLayer)−min(BaseLayer)
5. Reconstruct the luminance: D' = 10^(𝛾*BaseLayer + DetailLayer)
6. Compute D = D' *          1/
                     10^(max(𝛾*BaseLayer))
</pre>

**Command:**  
<pre>
python3 tone_mapping.py /PATH/TO/INPUT/IMAGE --op log/durand/all
i.e. python3 tone_mapping.py ./test_images/doll.hdr --op all
</pre>