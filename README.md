# i16_peakfit
Wrapper and graphical user interface of lmfit for scattering experiments such as those on Diamond-I16

### Usage
Start GUI session:
``` bash
  $ python -m i16_peakfit
 ```
OR - Start GUI with data loaded
``` bash
  $ python -m i16_peakfit /some/xye/file.dat
```

![GUI screenshot](i16_peakfit_screenshot.png)

OR - Import functions into script
``` python
    from i16_peakfit import multipeakfit, load_xye, peak_results_str
    
    xdata, ydata, yerror = load_xye('/some/data/file.txt')
    res = multipeakfit(x, y, npeaks=None, model='Gauss', plot_result=True)
    print(peak_results_str(res))
```

By Dan Porter, Diamond Light Source Ltd. 2021


