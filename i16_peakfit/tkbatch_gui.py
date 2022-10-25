"""
tkinter batch processing gui
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
import tkinter as tk

import i16_peakfit.functions
from i16_peakfit import fitting
from i16_peakfit import tk_widgets
from i16_peakfit.tk_widgets import (TF, BF, SF, HF, MF, LF,
                                    bkg, ety, btn, btn2, opt, btn_active, opt_active, txtcol,
                                    ety_txt, btn_txt, opt_txt, ttl_txt)
from i16_peakfit.tkinter_gui import FittingGUI
from i16_peakfit.tkmodelpars import ModelParsGUI
from i16_peakfit.peak_finding import find_peaks

# Figure size
_figure_size = [8, 4]
_figure_dpi = 40


def load_files():
    filenames = tk_widgets.filedialog.askopenfilenames(
        title='Select Files for batch processing',
        initialdir='.',
        initialfile='data.dat',
        defaultextension='*.*',
        filetypes=(("all files", "*.*"), ("text files", "*.txt"), ("data files", "*.dat"), ("CSV files", "*.csv"))
    )

    datasets = []
    for filename in filenames:
        xdata, ydata, error = i16_peakfit.functions.load_xye(filename)
        datasets += [(xdata, ydata, error)]
    return datasets


class DataLabelGui:
    """
    A Small Gui to capture dataset variables
    """

    def __init__(self, parent, dataset_values, name='Value'):
        """Initialise"""

        # Create Tk inter instance
        self.parent = parent  # i16_peakfit.tkinter_gui.BatchGui()
        self.root = tk.Toplevel(self.parent.root)
        self.root.wm_title('Dataset Values')
        self.root.minsize(width=100, height=300)
        self.root.maxsize(width=1200, height=1200)
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        self.value_name = tk.StringVar(self.root, name)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)

        var = tk.Label(frame, text='Datasets: %d' % len(dataset_values), font=TF)
        var.pack(side=tk.LEFT, padx=2)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        var = tk.Label(frame, text='Name: ', font=TF)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(frame, textvariable=self.value_name, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)

        sec = tk.LabelFrame(self.root, text='Values', relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=2, pady=2)
        self.txt_v = tk.Text(sec, width=65, wrap=tk.WORD, height=5, bg=ety)
        self.txt_v.insert(tk.INSERT, str(list(dataset_values)))
        scl = tk.Scrollbar(sec)
        scl.config(command=self.txt_v.yview)
        self.txt_v.config(yscrollcommand=scl.set)
        self.txt_v.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=0)
        scl.pack(side=tk.LEFT, fill=tk.Y)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        var = tk.Button(frame, text='Update', font=BF, command=self.but_update,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=2)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def but_update(self):
        """update parent & exit"""
        name = self.value_name.get()
        values = list(eval(self.txt_v.get('1.0', tk.END)))
        self.parent.set_dataset_values(values, name)
        self.f_exit()

    def f_exit(self):
        self.root.destroy()


class DataSetFrame:
    """
    A standalone GUI for Fitting
    """

    def __init__(self, parent, xdata=None, ydata=None, yerrors=None, name='Dataset', value=1):
        """Initialise"""

        if xdata is None:
            xdata = range(11)

        if ydata is None:
            ydata = i16_peakfit.functions.gauss(xdata, cen=np.mean(xdata), fwhm=np.ptp(xdata) / 4, height=1000)
        ydata = np.asarray(ydata, dtype=float)

        if yerrors is None:
            yerrors = i16_peakfit.functions.error_func(ydata)

        # Create tkinter Frame
        self.parent = parent  # BatchGui()
        self.root = tk.LabelFrame(parent.dataset_frame, text='Dataset')
        self.root.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # variables
        self.dataset_name = tk.StringVar(self.root, name)
        self.dataset_value = tk.StringVar(self.root, str(value))

        # Plotting lines
        self.mask = None
        self.ax1_component_lines = []
        # Models
        self.model = None
        self.pars = None
        # Results
        self.results = None

        # ----------------------------------------------------------------------------------------
        # ---Left hand side - Data Input---
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, expand=tk.YES)

        dst = tk.Frame(left)
        dst.pack(side=tk.LEFT, fill=tk.Y)

        # ------- X, Y, Error Data --------
        sec = tk.Frame(dst)
        sec.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=2, pady=2)
        # Dataset value
        frm = tk.Frame(sec)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        var = tk.Label(frm, textvariable=self.dataset_name, width=12, font=SF)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(frm, textvariable=self.dataset_value, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        # X
        self.txt_x = tk.Text(sec, width=65, height=1, bg=ety, wrap=tk.NONE)
        self.txt_x.insert(tk.INSERT, str(list(xdata)))
        self.txt_x.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=0)
        # Y
        self.txt_y = tk.Text(sec, width=65, height=1, bg=ety, wrap=tk.NONE)
        self.txt_y.insert(tk.INSERT, str(list(ydata)))
        self.txt_y.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=0)
        # Error
        self.txt_e = tk.Text(sec, width=65, height=1, bg=ety, wrap=tk.NONE)
        self.txt_e.insert(tk.INSERT, str(list(yerrors)))
        self.txt_e.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=0)

        scl = tk.Scrollbar(sec, orient='horizontal')

        def scroll(*args):
            self.txt_x.xview(*args)
            self.txt_y.xview(*args)
            self.txt_e.xview(*args)

        def fun_move(*args):
            """Move within text frame"""
            scl.set(*args)
            self.txt_x.yview('moveto', args[0])
            self.txt_y.yview('moveto', args[0])
            self.txt_e.yview('moveto', args[0])

        scl.config(command=scroll)
        self.txt_x.config(xscrollcommand=fun_move)
        self.txt_y.config(xscrollcommand=fun_move)
        self.txt_e.config(xscrollcommand=fun_move)
        scl.pack(side=tk.TOP, fill=tk.X)

        # ---Left hand side Middle - Buttons ---
        mid = tk.Frame(sec)
        mid.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)

        var = tk.Button(mid, text='Fitting\nGUI', font=BF, command=self.but_fitting_gui,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=2)
        var = tk.Button(mid, text='Models\n & \nParameters', font=BF, command=self.but_model_pars,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=2)

        # ---Right hand side Top - Buttons ---
        sec = tk.Frame(mid)
        sec.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=2, pady=2)

        frm = tk.Frame(sec)
        frm.pack(side=tk.TOP, fill=tk.Y, expand=tk.YES)
        var = tk.Button(frm, text='Peaks', font=BF, command=self.but_find_peaks,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Fit', font=BF, command=self.but_fit,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Copy to all', font=BF, command=self.but_copy,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Remove', font=BF, command=self.but_remove,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        frm = tk.Frame(sec)
        frm.pack(side=tk.TOP, fill=tk.Y, expand=tk.YES)
        var = tk.Button(frm, text='Display Results', font=BF, command=self.but_txt_results,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Plot Results', font=BF, command=self.but_plot_results,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # ------------------------------
        #     ---Right hand side---
        # ------------------------------
        right = tk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)

        # ---Right hand Middle - figure---
        # Figure - Data and Fit
        frm = tk.Frame(right)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=8)

        self.root.update_idletasks()
        figure_dpi = left.winfo_height() / _figure_size[1]
        self.fig1 = Figure(figsize=_figure_size, dpi=figure_dpi)
        self.fig1.patch.set_facecolor('w')

        canvas = FigureCanvasTkAgg(self.fig1, frm)
        canvas.get_tk_widget().configure(bg='black')
        canvas.draw()

        self.ax1 = self.fig1.add_subplot(111)
        self.pl_mask, = self.ax1.plot([], [], 'b+', lw=2, ms=6, label='Masked Points')
        self.pl_data, = self.ax1.plot(xdata, ydata, 'b-o', lw=2, ms=6, label='Data')
        self.pl_fit, = self.ax1.plot([], [], 'r-', lw=2, ms=6, label='Fit')
        self.pl_span = self.ax1.axvspan(np.min(xdata), np.min(xdata), alpha=0.2)
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
        #canvas.mpl_connect('button_press_event', self.plot_click)

        # Toolbar
        # frm = tk.Frame(right)
        # frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        # self.toolbar = NavigationToolbar2TkAgg(canvas, frm)
        # self.toolbar.update()
        # self.toolbar.pack(fill=tk.X, expand=tk.YES, padx=10)

        # ---Right hand bottom - Results textbox---
        sec = tk.LabelFrame(right, text='Results', relief=tk.RIDGE)
        #sec.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=2, pady=2)
        self.txt_res = tk.Text(sec, width=30, wrap=tk.WORD, height=5, bg=ety)
        self.txt_res.insert(tk.INSERT, '')
        scl = tk.Scrollbar(sec)
        scl.config(command=self.txt_res.yview)
        self.txt_res.config(yscrollcommand=scl.set)
        self.txt_res.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=1)
        scl.pack(side=tk.LEFT, fill=tk.Y)

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"
    def update_plot(self):
        """Update plot"""
        self.ax1.relim()
        self.ax1.autoscale(True)
        self.ax1.autoscale_view()
        self.fig1.canvas.draw()
        # self.toolbar.update()

    def update(self):
        xdata, ydata, yerror = self.gen_data()
        if self.mask is not None and len(self.mask) == len(xdata):
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
            yerror = yerror[~self.mask]

        # Update plot
        self.plot_data()
        if self.model is None:
            return
        xinit = np.linspace(xdata.min(), xdata.max(), xdata.size * 10)
        yinit = self.model.eval(params=self.pars, x=xinit)
        self.pl_fit.set_xdata(xinit)
        self.pl_fit.set_ydata(yinit)
        self.update_plot()

    def update_model(self, model, pars):
        """Add new model"""
        self.model = model
        self.pars = pars
        self.update()

    def set_data(self, xdata, ydata, error=None):
        """Replace txt boxes with new data"""
        if error is None:
            error = i16_peakfit.functions.error_func(ydata)

        self.txt_x.delete('1.0', tk.END)
        self.txt_x.insert('1.0', str(list(xdata)))
        self.txt_y.delete('1.0', tk.END)
        self.txt_y.insert('1.0', str(list(ydata)))
        self.txt_e.delete('1.0', tk.END)
        self.txt_e.insert('1.0', str(list(error)))
        self.plot_data()

    def gen_data(self):
        """Generate xdata, ydata, yerror"""
        xdata = np.array(eval(self.txt_x.get('1.0', tk.END)), dtype=float)
        ydata = np.array(eval(self.txt_y.get('1.0', tk.END)), dtype=float)
        yerror = np.array(eval(self.txt_e.get('1.0', tk.END)), dtype=float)
        return xdata, ydata, yerror

    def plot_data(self):
        """Get data and add to plot"""
        xdata, ydata, yerror = self.gen_data()
        xmin = np.min(xdata)
        xmask, ymask = [], []
        if self.mask is not None:
            xmask = xdata[self.mask]
            ymask = ydata[self.mask]
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
        self.pl_data.set_xdata(xdata)
        self.pl_data.set_ydata(ydata)
        self.pl_mask.set_xdata(xmask)
        self.pl_mask.set_ydata(ymask)
        ax_ymin, ax_ymax = self.ax1.get_ylim()
        span = [[xmin, ax_ymin], [xmin, ax_ymax], [xmin, ax_ymax], [xmin, ax_ymin], [xmin, ax_ymin]]
        self.pl_span.set_xy(span)
        self.update_plot()

    def find_peaks(self):
        """Button Find Peaks"""
        xdata, ydata, yerror = self.gen_data()
        if self.mask is not None and len(self.mask) == len(xdata):
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
            yerror = yerror[~self.mask]

        # get parameters from parent
        npeaks = self.parent.npeaks.get() if self.parent.npeaks_button.get() else None
        power = self.parent.min_peak_power.get() if self.parent.min_power_button.get() else None
        points = self.parent.points_between_peaks.get()
        model_name = self.parent.peak_model.get()
        bkg_name = self.parent.bkg_model.get()

        # Run find peaks, create message
        idx, pwr = find_peaks(ydata, yerror, min_peak_power=power, points_between_peaks=points)
        s = "Found %d peaks\n" % len(idx)
        s += "  Index | Position | Power\n"
        for n in range(len(idx)):
            s += "  %5d | %8.4g | %5.2f\n" % (idx[n], xdata[idx[n]], pwr[n])
        self.txt_res.delete('1.0', tk.END)
        self.txt_res.insert('1.0', s)

        # Create model (re-run find peaks)
        mod, pars = fitting.generate_model(
            xvals=xdata,
            yvals=ydata,
            yerrors=yerror,
            model=model_name,
            background=bkg_name,
            npeaks=npeaks,
            min_peak_power=power,
            points_between_peaks=points
        )
        # Replace models
        self.update_model(mod, pars)

    def add_peak(self, peak_type, xpos, ypos):
        """Add new peak model"""
        new_prefix = fitting.new_peak_prefix(self.model)
        new_model = fitting.getmodel(peak_type)(prefix=new_prefix)
        new_pars = new_model.make_params()

        # Set model parameters
        xdata, ydata, yerror = self.gen_data()
        sigma = (np.max(xdata) - np.min(xdata)) / 5
        amp = sigma * ypos / 0.3989423  # height = 0.3989423*amplitude/sigma
        pkcen = '%scenter' % new_prefix
        pkamp = '%samplitude' % new_prefix
        pksig = '%ssigma' % new_prefix
        new_pars[pkamp].set(value=amp, min=0, max=np.inf)
        new_pars[pkcen].set(value=xpos, min=np.min(xdata), max=np.max(xdata))
        new_pars[pksig].set(value=sigma, min=np.min(np.abs(np.diff(xdata))), max=3 * sigma)

        if self.model is None:
            self.model = new_model
            self.pars = new_pars
        else:
            self.model += new_model
            self.pars.update(new_pars)
        self.update()

    def run_fit(self):
        # Get data
        xdata, ydata, yerror = self.gen_data()
        if self.mask is not None and len(self.mask) == len(xdata):
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
            yerror = yerror[~self.mask]
        # Get model
        method = self.parent.fit_method.get()
        weights = i16_peakfit.functions.gen_weights(yerror)
        # Perform fit
        res = self.model.fit(ydata, self.pars, x=xdata, weights=weights, method=method)
        self.results = res
        self.update_model(res.model, res.params)

        # Update plot
        self.plot_data()
        xfit, yfit = fitting.peak_results_fit(res)
        self.pl_fit.set_xdata(xfit)
        self.pl_fit.set_ydata(yfit)
        self.update_plot()

        # Update results
        res_str = fitting.peak_results_str(res)
        self.txt_res.delete('1.0', tk.END)
        self.txt_res.insert('1.0', res_str)

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def but_fitting_gui(self):
        """Button Fitting GUI"""
        xdata, ydata, yerror = self.gen_data()
        FittingGUI(
            xdata=xdata,
            ydata=ydata,
            yerrors=yerror,
            model=self.model,
            parameters=self.pars,
            batch_parent=self
        )

    def but_find_peaks(self):
        """Button find peaks"""
        self.find_peaks()

    def but_sel_peaks(self):
        """Button Select Peaks"""

        def get_mouseposition(event):
            self.fig1.canvas.mpl_disconnect(press)
            # self.root.unbind("<Button-1>")
            self.root.config(cursor="arrow")

            if event.inaxes:
                # Add peak
                self.add_peak('Gaussian', event.xdata, event.ydata)

        press = self.fig1.canvas.mpl_connect('button_press_event', get_mouseposition)
        # self.root.bind("<Button-1>", get_mouseposition)
        self.root.config(cursor="crosshair")

    def but_click_select(self):
        """Button click to select"""

        xdata, ydata, yerror = self.gen_data()
        if self.mask is None or len(ydata) != len(self.mask):
            self.mask = np.zeros_like(ydata).astype(bool)

        xval = [np.min(xdata)]
        ipress = [False]

        def dissconect():
            self.plot_data()
            self.fig1.canvas.mpl_disconnect(press)
            self.fig1.canvas.mpl_disconnect(move)
            self.fig1.canvas.mpl_disconnect(release)
            self.root.config(cursor="arrow")

        def mouse_press(event):
            if event.inaxes:
                xval[0] = event.xdata
                ipress[0] = True
            else:
                dissconect()

        def mouse_move(event):
            if event.inaxes and ipress[0]:
                x_max = event.xdata
                ax_ymin, ax_ymax = self.ax1.get_ylim()
                span = [[xval[0], ax_ymin], [xval[0], ax_ymax], [x_max, ax_ymax], [x_max, ax_ymin], [xval[0], ax_ymin]]
                self.pl_span.set_xy(span)
                self.update_plot()

        def mouse_release(event):
            x_max = event.xdata
            x_min = xval[0]
            new_selection = (xdata > x_min) * (xdata < x_max)
            if np.any(self.mask[new_selection]):
                self.mask[new_selection] = False
            else:
                self.mask[~new_selection] = True
            dissconect()

        self.plot_data()
        press = self.fig1.canvas.mpl_connect('button_press_event', mouse_press)
        move = self.fig1.canvas.mpl_connect('motion_notify_event', mouse_move)
        release = self.fig1.canvas.mpl_connect('button_release_event', mouse_release)
        # self.root.bind("<Button-1>", get_mouseposition)
        self.root.config(cursor="crosshair")

    def but_click_mask(self):
        """Button click to mask"""

        xdata, ydata, yerror = self.gen_data()
        if self.mask is None or len(ydata) != len(self.mask):
            self.mask = np.zeros_like(ydata).astype(bool)

        xval = [np.min(xdata)]
        ipress = [False]

        def dissconect():
            self.plot_data()
            self.fig1.canvas.mpl_disconnect(press)
            self.fig1.canvas.mpl_disconnect(move)
            self.fig1.canvas.mpl_disconnect(release)
            self.root.config(cursor="arrow")

        def mouse_press(event):
            if event.inaxes:
                xval[0] = event.xdata
                ipress[0] = True
            else:
                dissconect()

        def mouse_move(event):
            if event.inaxes and ipress[0]:
                x_max = event.xdata
                ax_ymin, ax_ymax = self.ax1.get_ylim()
                span = [[xval[0], ax_ymin], [xval[0], ax_ymax], [x_max, ax_ymax], [x_max, ax_ymin], [xval[0], ax_ymin]]
                self.pl_span.set_xy(span)
                self.update_plot()

        def mouse_release(event):
            x_max = event.xdata
            x_min = xval[0]
            new_selection = (xdata > x_min) * (xdata < x_max)
            self.mask[new_selection] = True
            dissconect()

        self.plot_data()
        press = self.fig1.canvas.mpl_connect('button_press_event', mouse_press)
        move = self.fig1.canvas.mpl_connect('motion_notify_event', mouse_move)
        release = self.fig1.canvas.mpl_connect('button_release_event', mouse_release)
        # self.root.bind("<Button-1>", get_mouseposition)
        self.root.config(cursor="crosshair")

    def but_model_pars(self):
        """Button Models & Parameters"""
        ModelParsGUI(self, self.model, self.pars)

    def but_update(self):
        """Button update plot"""
        self.update()

    def but_fit(self):
        """Button fit"""
        self.run_fit()

    def but_txt_results(self):
        """Button Display Results"""
        if self.results is None:
            return
        out = fitting.peak_results_str(self.results)
        tk_widgets.StringViewer(out, 'i16_peakfit')

    def but_plot_results(self):
        """Button Plot Results"""
        if self.results is None:
            return
        fitting.peak_results_plot(self.results)

    def but_copy(self):
        """Button copy"""
        for dataset in self.parent.datasets:
            dataset.update_model(self.model, self.pars)

    def but_remove(self):
        """Button remove"""
        idx = self.parent.datasets.index(self)
        self.root.destroy()
        del self.parent.datasets[idx]


class BatchGui:
    """
    A GUI for batch processing of multiple sets of 1D data

    Usage:
        datasets = ((xdata, ydata, yerror), (xdata2, ydata2, yerror2))
        BatchGui(datasets)
    """

    def __init__(self, datasets=None, dataset_values=None, value_name='Value'):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('I16 Batch Fitting')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        self.datasets = []
        self.value_name = value_name

        self.npeaks = tk.IntVar(self.root, 1)
        self.npeaks_button = tk.BooleanVar(self.root, False)
        self.min_peak_power = tk.DoubleVar(self.root, None)
        self.min_power_button = tk.BooleanVar(self.root, False)
        self.points_between_peaks = tk.DoubleVar(self.root, 6)
        self.fit_method = tk.StringVar(self.root, 'leastsqr')
        self.peak_model = tk.StringVar(self.root, 'Gaussian')
        self.bkg_model = tk.StringVar(self.root, 'Linear')

        # Create tkinter Frame
        #frame = tk.Frame(self.root)
        #frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)

        # ------ Menu -------
        menu = {
            'File': {
                'New Window': self.menu_new,
                'Load Data': self.menu_load,
                'Load Nexus': self.menu_nexus,
                'Exit': self.f_exit,
            },
            'Fit': {
                'Display Output': self.menu_results_txt,
                'Plot Fit': self.but_plot_results,
                'Save results as JSON': self.menu_save_results_json,
                'Save results as CSV': self.menu_save_results_txt
            },
            'Script': {
                'Create Script': self.menu_script,
                'Create LMFit Script': self.menu_lmfit_script,
                'Open Script': self.menu_script_window,
            },
            'Help': {
                'About': tk_widgets.popup_about,
                'Help': tk_widgets.popup_help,
                'Examples': self.menu_examples,
                'Documentation': self.menu_docs,
                'GitHub Page': self.menu_github,
            }
        }
        tk_widgets.topmenu(self.root, menu)

        # ---------- Top Buttons ----------
        sec = tk.Frame(self.root)
        sec.pack(side=tk.TOP, fill=tk.X)

        frm = tk.Frame(sec)
        frm.pack(side=tk.LEFT, fill=tk.Y)
        var = tk.Button(frm, text='Add Dataset', font=BF, command=self.but_add_dataset,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, padx=2)
        var = tk.Button(frm, text='Load File', font=BF, command=self.but_load_file,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, padx=2)

        frm = tk.Frame(sec)
        frm.pack(side=tk.LEFT, fill=tk.Y)
        var = tk.Button(frm, text='Set dataset values', font=BF, command=self.but_dataset_values,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)
        var = tk.Button(frm, text='Fit all', font=BF, command=self.but_fit_all,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)
        var = tk.Button(frm, text='Plot Results', font=BF, command=self.but_plot_results,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)

        # Find peaks
        frm = tk.LabelFrame(sec, text='Find Peaks')
        frm.pack(side=tk.LEFT, fill=tk.Y)
        var = tk.Button(frm, text='Find Peaks', font=BF, width=12, command=self.but_find_peaks,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        fm1 = tk.Frame(frm)
        fm1.pack(side=tk.LEFT, fill=tk.Y)
        fm2 = tk.Frame(fm1)
        fm2.pack(fill=tk.X, expand=tk.YES)
        var = tk.Label(fm2, text='npeaks:', font=TF, width=12, anchor="e")
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(fm2, textvariable=self.npeaks, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Checkbutton(fm2, variable=self.npeaks_button)
        var.pack(side=tk.LEFT, padx=1)
        fm2 = tk.Frame(fm1)
        fm2.pack(fill=tk.X, expand=tk.YES)
        var = tk.Label(fm2, text='min power:', font=TF, width=12, anchor="e")
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(fm2, textvariable=self.min_peak_power, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Checkbutton(fm2, variable=self.min_power_button)
        var.pack(side=tk.LEFT, padx=1)
        fm2 = tk.Frame(fm1)
        fm2.pack(fill=tk.X, expand=tk.YES)
        var = tk.Label(fm2, text='min distance:', font=TF, width=12, anchor="e")
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(fm2, textvariable=self.points_between_peaks, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)

        # Model Options
        fm1 = tk.Frame(frm)
        fm1.pack(side=tk.LEFT, fill=tk.Y)
        models = [mod.capitalize() for mod in fitting.PEAK_MODELS if mod in fitting.MODELS]
        var = tk.OptionMenu(fm1, self.peak_model, *models)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.TOP, fill=tk.X)
        models = [mod.capitalize() for mod in fitting.BACKGROUND_MODELS if mod in fitting.MODELS]
        var = tk.OptionMenu(fm1, self.bkg_model, *models)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.TOP, fill=tk.X)
        methods = list(fitting.METHODS.keys())
        var = tk.OptionMenu(fm1, self.fit_method, *methods)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.TOP, fill=tk.X)

        "------------------------ Scollable Canvas ----------------------------------"
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        can = tk.Canvas(frame, height=600)
        #self.canvas = can
        scl = tk.Scrollbar(frame, orient=tk.VERTICAL, command=can.yview)

        self.canvas_frame = tk.Frame(can)
        self.canvas_frame.bind(
            "<Configure>",
            lambda e: can.configure(
                scrollregion=can.bbox("all")
            )
        )
        can.create_window((0, 0), window=self.canvas_frame, anchor="nw")
        can.configure(yscrollcommand=scl.set)

        def _on_mousewheel(event):
            can.yview_scroll(int(-1 * (event.delta / 60)), "units")

        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        can.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scl.pack(side=tk.RIGHT, fill=tk.Y)

        # ---------- Datasets ----------
        self.dataset_frame = tk.LabelFrame(self.canvas_frame, text='Datasets', relief=tk.RIDGE)
        self.dataset_frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)

        if datasets is None:
            self.load_files()
        else:
            if np.ndim(datasets) < 2:
                datasets = np.reshape(datasets, (-1, len(datasets)))
            if dataset_values is None:
                dataset_values = range(1, len(datasets)+1)

            for dataset, value in zip(datasets, dataset_values):
                xdata, ydata, error = dataset
                self.add_dataset(xdata, ydata, error, value)

        self.dataset_frame.update_idletasks()
        can.configure(width=self.dataset_frame.winfo_width())

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def load_files(self):
        filenames = tk_widgets.filedialog.askopenfilenames(
            title='Select Files for batch processing',
            initialdir='.',
            initialfile='data.dat',
            defaultextension='*.*',
            filetypes=(("all files", "*.*"), ("text files", "*.txt"), ("data files", "*.dat"), ("CSV files", "*.csv"),
                       ("Nexus files", "*.nxs"), ("HDF file", "*.hdf5"))
        )

        if filenames is None:
            return

        if filenames[0].endswith('nxs') or filenames[0].endswith('hdf5'):
            from i16_peakfit.tknexus_selector import NexusSelectorGui

            xdata, ydata, error, values, batch_name = NexusSelectorGui(self, filenames).show()
            self.value_name = batch_name
            for n in range(len(xdata)):
                self.add_dataset(list(xdata[n]), list(ydata[n]), list(error[n]), values[n])
        else:
            for filename in filenames:
                xdata, ydata, error = i16_peakfit.functions.load_xye(filename)
                self.add_dataset(xdata, ydata, error)

    def add_dataset(self, xdata, ydata, yerror, value=None):
        new_dataset_number = len(self.datasets) + 1
        if value is None:
            value = new_dataset_number
        name = 'Dataset %d' % new_dataset_number
        new_dataset = DataSetFrame(self, xdata, ydata, yerror, name, value)
        self.datasets += [new_dataset]

    def get_dataset_values(self):
        """Get dataset values"""
        values = []
        for dataset in self.datasets:
            values += [eval(dataset.dataset_value.get())]
        return values

    def set_dataset_values(self, values, name=None):
        """Set dataset values"""
        if name is not None:
            self.value_name = name

        for n, dataset in enumerate(self.datasets):
            dataset.dataset_name.set('Dataset %d' % (n+1))
            dataset.dataset_value.set(str(values[n]))

    def get_results(self):
        """Get results from fitted models"""
        return [fitting.peak_results(dataset.results) if dataset.results else {} for dataset in self.datasets]

    def get_result(self, name):
        """Return a particular result, e.g. amplitude, center, sigma, height, fwhm, background"""
        results = self.get_results()
        return [result[name] if name in result else np.nan for result in results]

    def save_results_json(self, filename):
        from i16_peakfit.functions import save_results_json
        values = self.get_dataset_values()
        results = {value: fitting.peak_results(dataset.results) if dataset.results else {} for value, dataset in
                   zip(values, self.datasets)}
        save_results_json(results, filename)

    def get_results_txt(self):
        """Generate batch results as str"""
        values = self.get_dataset_values()
        results = self.get_results()
        head_vals = ['amplitude', 'height', 'center', 'sigma', 'fwhm', 'background']
        head = "%s, " % self.value_name
        head += ", ".join(head_vals)
        result = {h: [result[h] if h in result else np.nan for result in results] for h in head_vals}
        out = '%s\n' % head
        fmt = '{%s}, ' % self.value_name
        fmt += ', '.join(['{%s:12.6g}' % h for h in head_vals])
        for n in range(len(values)):
            result = {h: results[n][h] if h in results[n] else np.nan for h in head_vals}
            result[self.value_name] = values[n]
            out += '%s\n' % fmt.format(**result)
        return out

    "------------------------------------------------------------------------"
    "-----------------------------Menu Functions-----------------------------"
    "------------------------------------------------------------------------"

    def menu_new(self):
        """Menu button new window"""
        BatchGui([(None, None, None)])

    def menu_load(self):
        """Menu button Load text"""
        self.load_files()

    def menu_nexus(self):
        """Menu button Load nexus"""
        self.load_files()

    def menu_save_results_json(self):
        filename = tk_widgets.filedialog.asksaveasfilename(
            title='Save Results JSON',
            initialdir='.',
            initialfile='results.json',
            defaultextension='*.json',
            filetypes=(("JSON files", "*.json"), ("all files", "*.*"))
        )
        if filename:
            self.save_results_json(filename)

    def menu_save_results_txt(self):
        filename = tk_widgets.filedialog.asksaveasfilename(
            title='Save Results CSV',
            initialdir='.',
            initialfile='results.csv',
            defaultextension='*.csv',
            filetypes=(("CSV files", "*.csv"), ('DAT files', '*.dat'), ('Text files', '*.txt'), ("all files", "*.*"))
        )
        if filename:
            s = self.get_results_txt()
            with open(filename, 'wt') as f:
                f.write(s)

    def menu_results_txt(self):
        s = self.get_results_txt()
        tk_widgets.StringViewer(s, 'i16_peakfit Batch results')

    def menu_script(self):
        """Menu button Create Script"""
        xdata, ydata, yerror = self.gen_data()
        if self.mask is not None and len(self.mask) == len(xdata):
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
            yerror = yerror[~self.mask]
        model_name = self.peak_model.get()
        bkg_name = self.bkg_model.get()
        script = fitting.generate_model_script(xdata, ydata, yerror, model=model_name, background=bkg_name)
        tk_widgets.PythonEditor(script, 'i16 peakfit')

    def menu_lmfit_script(self):
        """Menu button Create lmfit script"""
        xdata, ydata, yerror = self.gen_data()
        if self.mask is not None and len(self.mask) == len(xdata):
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
            yerror = yerror[~self.mask]
        model_name = self.peak_model.get()
        bkg_name = self.bkg_model.get()
        script = fitting.generate_model_script(xdata, ydata, yerror,
                                               model=model_name, background=bkg_name,
                                               include_i16_peakfit=False)
        tk_widgets.PythonEditor(script, 'lmfit')

    def menu_script_window(self):
        """Menu button Script Window"""
        newsavelocation = tk_widgets.filedialog.askopenfilename(
            title='Open a python script',
            initialdir='.',
            initialfile='script.py',
            defaultextension='.py',
            filetypes=(("python file", "*.py"), ("all files", "*.*"))
        )

        if newsavelocation == '':
            return
        with open(newsavelocation) as file:
            disp_str = file.read()
        tk_widgets.PythonEditor(disp_str, newsavelocation)

    def menu_examples(self):
        pass

    def menu_docs(self):
        pass

    def menu_github(self):
        pass

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def but_add_dataset(self):
        self.add_dataset(None, None, None)

    def but_load_file(self):
        self.load_files()

    def but_dataset_values(self):
        """Button set dataset values"""
        values = self.get_dataset_values()
        name = self.value_name
        DataLabelGui(self, values, name)

    def but_fit_all(self):
        for dataset in self.datasets:
            dataset.run_fit()

    def but_plot_results(self):
        from i16_peakfit.plotting import plot_batch_results
        values = self.get_dataset_values()
        results = self.get_results()
        value_name = self.value_name
        plot_batch_results(values, results, value_name)

    def but_find_peaks(self):
        """Button Find Peaks"""
        for dataset in self.datasets:
            dataset.find_peaks()

    def f_exit(self):
        self.root.destroy()


if __name__ == '__main__':
    # Run GUI
    gui = BatchGui()