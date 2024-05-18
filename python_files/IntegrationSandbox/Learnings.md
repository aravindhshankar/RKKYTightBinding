### WHAT DID WE LEARN? 
From the the native integration of 1/\sqrt(x) and log(x), both of which have integrable singularities, scipy.quad() is unable to perform the integration across the singularity, but is able to do really really well if the integral is split up into I1 of the left of the singularity and I2 on the right of the singularity. 

But the position of the singularity must be EXACTLY known, even a little bit of error there leads quad to fail.
