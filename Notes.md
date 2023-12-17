# SOme comments #

Good the idea of using the fastest Fourier transform in the West (fftw) to test your implementation! It is indeed the library integrated in Matlab.

The code looks very sophisticated and with use of several complex c++ features as well! Very good. You are using the Eigen library, which is a very good choice. 
However, since you are using unsupported features of Eigen (the Tensor module), you should clearly indicate in the README file the exact Eigen release or have included the Eigen library in your project (it's a header only library, so it's easy to do).

Nice having a documentation with Doxygen. However, it is very scarce and not very useful. It is ok for the scope of this hands-on, but if you want to write a more complete documentation, you should add more comments to the code and maybe use a personalised Doxygen file.

Nice also the description in the file in the doc folder (even if still incomplete).

In the presentation focus on the salient aspects of your code and the results you obtained. You can also show some code, but not too much. You can also show some results, but not too much. The presentation should be 15 minutes long, so you have to be very selective on what to show.

I have made just a comment in the code, marked with @note. 

