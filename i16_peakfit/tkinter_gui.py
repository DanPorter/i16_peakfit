"""
tkinter gui
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
from i16_peakfit.tkmodelpars import ModelParsGUI
from i16_peakfit.peak_finding import find_peaks

# Figure size
_figure_dpi = 60


class FittingGUI:
    """
    A standalone GUI for Fitting
    """

    def __init__(self, xdata=None, ydata=None, yerrors=None, model=None, parameters=None, batch_parent=None):
        """Initialise"""

        self.batch_parent = batch_parent  # /batch_gui.DataSetFrame()

        # Create Tk inter instance
        if self.batch_parent is None:
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel(self.batch_parent.root)
        self.root.wm_title('I16 Peak Fitting')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        if xdata is None:
            xdata = range(11)

        if ydata is None:
            ydata = i16_peakfit.functions.gauss(xdata, cen=np.mean(xdata), fwhm=np.ptp(xdata) / 4, height=1000)
        ydata = np.asarray(ydata, dtype=float)

        if yerrors is None:
            yerrors = i16_peakfit.functions.error_func(ydata)

        # Create tkinter Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # variables
        self.message = tk.StringVar(self.root, "Start by adding data, then 'Select Peaks' to create a model")
        self.npeaks = tk.IntVar(self.root, 1)
        self.npeaks_button = tk.BooleanVar(self.root, False)
        self.min_peak_power = tk.DoubleVar(self.root, None)
        self.min_power_button = tk.BooleanVar(self.root, False)
        self.points_between_peaks = tk.DoubleVar(self.root, 6)
        self.fit_method = tk.StringVar(self.root, 'leastsqr')
        self.peak_model = tk.StringVar(self.root, 'Gaussian')
        self.bkg_model = tk.StringVar(self.root, 'Linear')

        # Plotting lines
        self.mask = None
        self.ax1_component_lines = []
        # Models
        if model is not None and parameters is None:
            parameters = model.make_params()
        self.model = model
        self.pars = parameters
        # Results
        self.results = None

        # ----------------------------------------------------------------------------------------
        # ---Menu---
        menu = {
            'File': {
                'New Window': self.menu_new,
                'Load Data': self.menu_load,
                'Load Nexus': self.menu_nexus,
                'Load Batch Data': self.menu_load_batch,
                'Exit': self.f_exit,
            },
            'Fit': {
                'Display Output': self.but_txt_results,
                'Plot Fit': self.but_plot_results,
                'Batch Fit': self.menu_batch,
            },
            'Script': {
                'Create Script': self.menu_script,
                'Create LMFit Script': self.menu_lmfit_script,
                'Open Script': self.menu_script_window,
            },
            'Help': {
                'Help': self.menu_help,
                'Examples': self.menu_examples,
                'Documentation': self.menu_docs,
                'GitHub Page': self.menu_github,
            }
        }
        tk_widgets.topmenu(self.root, menu)

        # ----------------------------------------------------------------------------------------
        # ---Left hand side - Data Input---
        left = tk.Frame(frame)
        left.pack(side=tk.LEFT, expand=tk.YES)

        # ------- X, Y, Error Data --------
        # X
        sec = tk.LabelFrame(left, text='X Data', relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)
        self.txt_x = tk.Text(sec, width=65, wrap=tk.WORD, height=5, bg=ety)
        self.txt_x.insert(tk.INSERT, str(list(xdata)))
        scl = tk.Scrollbar(sec)
        scl.config(command=self.txt_x.yview)
        self.txt_x.config(yscrollcommand=scl.set)
        self.txt_x.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=0)
        scl.pack(side=tk.LEFT, fill=tk.Y)

        # Y
        sec = tk.LabelFrame(left, text='Y Data', relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)
        self.txt_y = tk.Text(sec, width=65, wrap=tk.WORD, height=5, bg=ety)
        self.txt_y.insert(tk.INSERT, str(list(ydata)))
        scl = tk.Scrollbar(sec)
        scl.config(command=self.txt_y.yview)
        self.txt_y.config(yscrollcommand=scl.set)
        self.txt_y.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=0)
        scl.pack(side=tk.LEFT, fill=tk.Y)

        # Error
        sec = tk.LabelFrame(left, text='Y Error', relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)
        self.txt_e = tk.Text(sec, width=65, wrap=tk.WORD, height=5, bg=ety)
        self.txt_e.insert(tk.INSERT, str(list(yerrors)))
        scl = tk.Scrollbar(sec)
        scl.config(command=self.txt_e.yview)
        self.txt_e.config(yscrollcommand=scl.set)
        self.txt_e.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=0)
        scl.pack(side=tk.LEFT, fill=tk.Y)

        # ---Left hand side Middle - Buttons ---
        mid = tk.Frame(left)
        mid.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)

        sec = tk.LabelFrame(mid, text='Find Peaks')
        sec.pack(side=tk.LEFT, fill=tk.Y)
        var = tk.Button(sec, text='Find Peaks', font=BF, width=12, command=self.but_find_peaks,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        frm = tk.Frame(sec)
        frm.pack(side=tk.LEFT, fill=tk.Y)
        fm2 = tk.Frame(frm)
        fm2.pack(fill=tk.X, expand=tk.YES)
        var = tk.Label(fm2, text='npeaks:', font=TF, width=12, anchor="e")
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(fm2, textvariable=self.npeaks, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Checkbutton(fm2, variable=self.npeaks_button)
        var.pack(side=tk.LEFT, padx=1)
        fm2 = tk.Frame(frm)
        fm2.pack(fill=tk.X, expand=tk.YES)
        var = tk.Label(fm2, text='min power:', font=TF, width=12, anchor="e")
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(fm2, textvariable=self.min_peak_power, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Checkbutton(fm2, variable=self.min_power_button)
        var.pack(side=tk.LEFT, padx=1)
        fm2 = tk.Frame(frm)
        fm2.pack(fill=tk.X, expand=tk.YES)
        var = tk.Label(fm2, text='min distance:', font=TF, width=12, anchor="e")
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(fm2, textvariable=self.points_between_peaks, font=TF, width=10, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)

        frm = tk.Frame(mid)
        frm.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Click to add Peak', font=BF, command=self.but_sel_peaks,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)
        var = tk.Button(frm, text='Click to Select', font=BF, command=self.but_click_select,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)
        var = tk.Button(frm, text='Click to Mask', font=BF, command=self.but_click_mask,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)

        frm = tk.Frame(mid)
        frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)
        var = tk.Button(frm, text='Models\n & \nParameters', font=BF, command=self.but_model_pars,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=2)

        # ---Left hand side Middle - Batch ---
        if batch_parent is not None:
            mid = tk.Frame(left)
            mid.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)

            var = tk.Button(frm, text='Update Batch', font=BF, command=self.but_batch_update,
                            bg=btn, activebackground=btn_active)
            var.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=2)

        # ------------------------------
        #     ---Right hand side---
        # ------------------------------
        right = tk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)

        # ---Right hand side top - Messagebox ---
        sec = tk.Frame(right)
        sec.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)

        var = tk.Label(sec, textvariable=self.message, font=SF)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

        # ---Right hand side Top - Buttons ---
        sec = tk.Frame(right)
        sec.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)
        var = tk.Button(sec, text='Fit', font=BF, width=12, command=self.but_fit,
                        bg=btn2, activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=tk.YES, padx=2)
        var = tk.Label(sec, text='Method:', font=SF)
        var.pack(side=tk.LEFT)
        methods = list(fitting.METHODS.keys())
        var = tk.OptionMenu(sec, self.fit_method, *methods)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(sec, text='Display Results', font=BF, command=self.but_txt_results,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=True, padx=2)
        var = tk.Button(sec, text='Plot Results', font=BF, command=self.but_plot_results,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, expand=True, padx=2)

        # ---Right hand Middle - figure---
        # Figure - Data and Fit
        frm = tk.Frame(right)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=8)

        self.fig1 = Figure(figsize=[16, 8], dpi=_figure_dpi)
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
        frm = tk.Frame(right)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        self.toolbar = NavigationToolbar2TkAgg(canvas, frm)
        self.toolbar.update()
        self.toolbar.pack(fill=tk.X, expand=tk.YES, padx=10)

        # ---Right hand bottom - Results textbox---
        sec = tk.LabelFrame(right, text='Results', relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=2, pady=2)
        self.txt_res = tk.Text(sec, width=30, wrap=tk.WORD, height=5, bg=ety)
        self.txt_res.insert(tk.INSERT, '')
        scl = tk.Scrollbar(sec)
        scl.config(command=self.txt_res.yview)
        self.txt_res.config(yscrollcommand=scl.set)
        self.txt_res.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES, padx=1)
        scl.pack(side=tk.LEFT, fill=tk.Y)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"
    def update_plot(self):
        """Update plot"""
        self.ax1.relim()
        self.ax1.autoscale(True)
        self.ax1.autoscale_view()
        self.fig1.canvas.draw()
        self.toolbar.update()

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
        method = self.fit_method.get()
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
        self.message.set('Fit completed.')

    "------------------------------------------------------------------------"
    "-----------------------------Menu Functions-----------------------------"
    "------------------------------------------------------------------------"

    def menu_new(self):
        """Menu button new window"""
        FittingGUI()

    def menu_load(self):
        """Menu button Load text"""
        filename = tk_widgets.filedialog.askopenfilename(
            title='Read (x y error) data',
            initialdir='.',
            initialfile='data.dat',
            defaultextension='*.*',
            filetypes=(("all files", "*.*"), ("text files", "*.txt"), ("data files", "*.dat"), ("CSV files", "*.csv"))
        )

        if filename == '':
            return

        self.pl_fit.set_xdata([])
        self.pl_fit.set_ydata([])
        self.pl_mask.set_xdata([])
        self.pl_mask.set_ydata([])
        self.results = None
        self.mask = None

        xdata, ydata, error = i16_peakfit.functions.load_xye(filename)
        self.set_data(xdata, ydata, error)

    def menu_nexus(self):
        """Menu button Load nexus"""
        from i16_peakfit.tknexus_selector import NexusSelectorGui, get_filenames

        files = get_filenames()
        if not files:
            return
        xdata, ydata, edata, values, batch_name = NexusSelectorGui(self, files[0]).show()

        self.pl_fit.set_xdata([])
        self.pl_fit.set_ydata([])
        self.pl_mask.set_xdata([])
        self.pl_mask.set_ydata([])
        self.results = None
        self.mask = None

        self.set_data(list(xdata[0]), list(ydata[0]), list(edata[0]))

    def menu_batch(self):
        """Menu button Start batch processing"""
        from i16_peakfit.tkbatch_gui import BatchGui
        xdata, ydata, yerror = self.gen_data()
        BatchGui([(xdata, ydata, yerror)])

    def menu_load_batch(self):
        """Menu button load files for batch processing"""
        from i16_peakfit.tkbatch_gui import BatchGui
        BatchGui()

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

    def menu_help(self):
        pass

    def menu_examples(self):
        pass

    def menu_docs(self):
        pass

    def menu_github(self):
        pass

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def but_find_peaks(self):
        """Button Find Peaks"""
        xdata, ydata, yerror = self.gen_data()
        if self.mask is not None and len(self.mask) == len(xdata):
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
            yerror = yerror[~self.mask]
        npeaks = self.npeaks.get() if self.npeaks_button.get() else None
        power = self.min_peak_power.get() if self.min_power_button.get() else None
        points = self.points_between_peaks.get()
        model_name = self.peak_model.get()
        bkg_name = self.bkg_model.get()

        # Run find peaks, create message
        idx, pwr = find_peaks(ydata, yerror, min_peak_power=power, points_between_peaks=points)
        s = "Found %d peaks\n" % len(idx)
        s += "  Index | Position | Power\n"
        for n in range(len(idx)):
            s += "  %5d | %8.4g | %5.2f\n" % (idx[n], xdata[idx[n]], pwr[n])
        self.message.set('Found %d peaks' % len(idx))
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

    def but_sel_peaks(self):
        """Button Select Peaks"""

        def get_mouseposition(event):
            print(event.x, event.y, event.xdata, event.ydata, event.inaxes)
            self.fig1.canvas.mpl_disconnect(press)
            # self.root.unbind("<Button-1>")
            self.root.config(cursor="arrow")

            if event.inaxes:
                self.message.set('Position selected: (%.4g, %.4g)' % (event.xdata, event.ydata))
                # Add peak
                self.add_peak('Gaussian', event.xdata, event.ydata)
            else:
                self.message.set('No position selected')

        self.message.set('Click on Figure to add a peak')
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
        self.message.set('Click & Drag region to select')
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
        self.message.set('Click & Drag region to select')
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

    def but_batch_update(self):
        """Button update batch (when batch_parent active)"""
        xdata, ydata, yerror = self.gen_data()
        self.batch_parent.set_data(xdata, ydata, yerror)
        self.batch_parent.mask = self.mask
        self.batch_parent.results = self.results
        self.batch_parent.update_model(self.model, self.pars)
        self.batch_parent.update()

    def f_exit(self):
        self.root.destroy()


if __name__ == '__main__':
    # Run GUI
    gui = FittingGUI()

