"""
tkinter gui
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
import tkinter as tk

from i16_peakfit import fitting
from i16_peakfit import tk_widgets
from .tk_widgets import (TF, BF, SF, HF, MF, LF,
                         bkg, ety, btn, btn2, opt, btn_active, opt_active, txtcol,
                         ety_txt, btn_txt, opt_txt, ttl_txt)

# Figure size
_figure_dpi = 60


def gen_param(parent, model, pars):
    """Make param section, adds section to TOP of parent, returns parameter boxes"""

    if hasattr(model, 'components'):
        comps = model.components
    else:
        comps = [model]

    mods = [(m.prefix, m._name) for m in comps]
    tkvars = {}
    sections = []
    for mod in comps:
        sec = tk.LabelFrame(parent, text=mod.name, relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=1, pady=1)
        sections += [sec]
        tkvars[mod.prefix] = []
        for par in pars:
            if mod.prefix not in par:
                continue
            pval = tk.DoubleVar(parent, pars[par].value)
            pmin = tk.DoubleVar(parent, pars[par].min)
            pmax = tk.DoubleVar(parent, pars[par].max)
            pref = tk.IntVar(parent, int(pars[par].vary))
            frm = tk.Frame(sec)
            frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
            var = tk.Label(frm, text='%s' % par, width=12, font=SF)
            var.pack(side=tk.LEFT, padx=1)
            var = tk.Label(frm, text='Value:', font=TF)
            var.pack(side=tk.LEFT, padx=2)
            var = tk.Entry(frm, textvariable=pval, font=TF, width=10, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT, padx=2)
            var = tk.Label(frm, text='Min:', font=TF)
            var.pack(side=tk.LEFT, padx=2)
            var = tk.Entry(frm, textvariable=pmin, font=TF, width=10, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT, padx=2)
            var = tk.Label(frm, text='Max:', font=TF)
            var.pack(side=tk.LEFT, padx=2)
            var = tk.Entry(frm, textvariable=pmax, font=TF, width=10, bg=ety, fg=ety_txt)
            var.pack(side=tk.LEFT, padx=2)
            var = tk.Label(frm, text='Refine:', font=TF)
            var.pack(side=tk.LEFT, padx=2)
            var = tk.Checkbutton(frm, variable=pref, font=TF)
            var.pack(side=tk.LEFT, padx=2)
            tkvars[mod.prefix] += [(par, pval, pmin, pmax, pref)]
    return mods, tkvars, sections


def gen_param_from_name(parent, model_name='Gaussian', prefix=''):
    """Make param section"""
    model = fitting.getmodel(model_name)(prefix=prefix)
    pars = model.make_params()
    return gen_param(parent, model, pars)


def gen_model_from_boxes(mods, tkvars):
    """Generate model from tkboxes"""
    models = []
    for prefix, modname in mods:
        # models += [fitting.MODELS[modname.lower()](prefix=prefix)]
        models += [fitting.getmodel(modname)(prefix=prefix)]
    model = models[0]
    for mod in models[1:]:
        model += mod
    pars = model.make_params()

    for prefix in tkvars:
        for param in tkvars[prefix]:
            par, pval, pmin, pmax, pref = param
            pars[par].set(value=pval.get(), min=pmin.get(), max=pmax.get(), vary=bool(pref.get()))
    return model, pars


class FittingGUI:
    """
    A standalone GUI for Fitting
    """

    def __init__(self, xdata=None, ydata=None, yerrors=None):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
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
            ydata = fitting.gauss(xdata, cen=np.mean(xdata), fwhm=np.ptp(xdata)/4, height=1000)
        ydata = np.asarray(ydata, dtype=float)

        if yerrors is None:
            yerrors = fitting.error_func(ydata)

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
        self.models = []  # [('bkg_', 'linear'), ('p1_', 'gaussian')]
        self.peakvars = {}  # {'bkg_': [('bkg_amplitude', tkvalue, tkmin, tkmax, tkvary), (bkg_slope,...]}
        self.par_sections = []  # [tk_LabelFrame('bkg_'), tk_LabelFrame('p1_') ...]
        # Results
        self.results = None

        # ----------------------------------------------------------------------------------------
        # ---Menu---
        menu = {
            'File': {
                'New Window': self.menu_new,
                'Load Data': self.menu_load,
                'Load Nexus': self.menu_nexus,
                'Exit': self.f_exit,
            },
            'Fit': {
                'Display Output': self.but_txt_results,
                'Plot Fit': self.but_plot_results,
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

        # ---Left hand side Middle - Messagebox ---
        mid = tk.Frame(left)
        mid.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)

        var = tk.Label(mid, textvariable=self.message, font=SF)
        var.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)

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

        var = tk.Button(mid, text='Select Peaks', font=BF, width=12, command=self.but_sel_peaks,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        frm = tk.Frame(mid)
        frm.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Click to Select', font=BF, command=self.but_click_select,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)
        var = tk.Button(frm, text='Click to Mask', font=BF, command=self.but_click_mask,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)

        # ---Left hand side Middle - Parameters ---
        mid = tk.LabelFrame(left, text='Model', relief=tk.RIDGE)
        mid.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2, pady=2)

        # Models
        frm = tk.Frame(mid)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.NO)
        models = [mod.capitalize() for mod in fitting.PEAK_MODELS]
        var = tk.OptionMenu(frm, self.peak_model, *models)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        models = [mod.capitalize() for mod in fitting.BACKGROUND_MODELS]
        var = tk.OptionMenu(frm, self.bkg_model, *models)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Add Peak', font=BF, command=self.but_add_peak,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Button(frm, text='Clear', font=BF, command=self.but_clear_peak,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Button(frm, text='Update', font=BF, command=self.but_update,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT, fill=tk.X, expand=tk.YES, padx=2)

        # Parameters
        sec = tk.LabelFrame(mid, text='Parameters', relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.BOTH, padx=2, pady=2)
        can = tk.Canvas(sec, height=200, width=900)
        scl = tk.Scrollbar(sec, orient=tk.VERTICAL, command=can.yview)
        self.peaksec = tk.Frame(can)

        self.peaksec.bind(
            "<Configure>",
            lambda e: can.configure(
                scrollregion=can.bbox("all")
            )
        )
        can.create_window((0, 0), window=self.peaksec, anchor="nw")
        can.configure(yscrollcommand=scl.set)

        def _on_mousewheel(event):
            can.yview_scroll(int(-1 * (event.delta / 60)), "units")
        can.bind("<MouseWheel>", _on_mousewheel)

        can.pack(side="left", fill="both", expand=True)
        scl.pack(side="right", fill="y")

        # self.add_model('Linear')

        # ------------------------------
        #     ---Right hand side---
        # ------------------------------
        right = tk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.Y, expand=tk.YES)

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
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

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
        self.toolbar.pack(fill=tk.X, expand=tk.YES)

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
    def add_model(self, name):
        """Add new model"""
        # Genereate prefix
        current_prefix = [mod[0] for mod in self.models]
        if name.lower() in fitting.BACKGROUND_MODELS:
            new_prefix = 'bkg_'
            if 'bkg_' in current_prefix:
                # remove old background + replace
                self.remove_model('bkg_')
        else:
            n = 1
            while 'p%d_' % n in current_prefix:
                n += 1
            new_prefix = 'p%d_' % n

        mods, tkvars, secs = gen_param_from_name(self.peaksec, name, new_prefix)
        self.models += mods
        self.peakvars.update(tkvars)
        self.par_sections += secs

    def remove_model(self, prefix):
        """Remove model by prefix"""
        current_prefix = [mod[0] for mod in self.models]
        if prefix not in current_prefix:
            return
        _ = self.peakvars.pop(prefix)
        idx = current_prefix.index(prefix)
        _ = self.models.pop(idx)
        frm = self.par_sections.pop(idx)
        frm.destroy()

    def remove_models(self):
        """Remove all models"""
        current_prefix = [mod[0] for mod in self.models]
        for prefx in current_prefix:
            # print('Removing %s' % prefx)
            self.remove_model(prefx)

    def gen_model(self):
        """Create model, pars from input boxes"""
        return gen_model_from_boxes(self.models, self.peakvars)  # mod, pars

    def gen_data(self):
        """Generate xdata, ydata, yerror"""
        xdata = np.array(eval(self.txt_x.get('1.0', tk.END)), dtype=float)
        ydata = np.array(eval(self.txt_y.get('1.0', tk.END)), dtype=float)
        yerror = np.array(eval(self.txt_e.get('1.0', tk.END)), dtype=float)
        return xdata, ydata, yerror

    def update_plot(self):
        """Update plot"""
        self.ax1.relim()
        self.ax1.autoscale(True)
        self.ax1.autoscale_view()
        self.fig1.canvas.draw()
        self.toolbar.update()

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

    def run_fit(self):
        # Get data
        xdata, ydata, yerror = self.gen_data()
        if self.mask is not None and len(self.mask) == len(xdata):
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
            yerror = yerror[~self.mask]
        # Get model
        model, pars = self.gen_model()
        method = self.fit_method.get()
        weights = fitting.gen_weights(yerror)
        # Perform fit
        res = model.fit(ydata, pars, x=xdata, weights=weights, method=method)
        self.results = res

        # update parameters
        for prefix in self.peakvars:
            for param in self.peakvars[prefix]:
                par, pval, pmin, pmax, pref = param
                pval.set(res.params[par].value)

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
    "---------------------------Button Functions-----------------------------"
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

        xdata, ydata, error = fitting.load_xye(filename)
        self.txt_x.delete('1.0', tk.END)
        self.txt_x.insert('1.0', str(list(xdata)))
        self.txt_y.delete('1.0', tk.END)
        self.txt_y.insert('1.0', str(list(ydata)))
        self.txt_e.delete('1.0', tk.END)
        self.txt_e.insert('1.0', str(list(error)))
        self.plot_data()

    def menu_nexus(self):
        """Menu button Load nexus"""
        pass

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
        idx, pwr = fitting.find_peaks(ydata, yerror, min_peak_power=power, points_between_peaks=points)
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
        self.remove_models()
        mods, tkvars, secs = gen_param(self.peaksec, mod, pars)
        self.models += mods
        self.peakvars.update(tkvars)
        self.par_sections += secs

        # Update plot
        self.plot_data()
        xinit = np.linspace(xdata.min(), xdata.max(), xdata.size * 10)
        yinit = mod.eval(params=pars, x=xinit)
        self.pl_fit.set_xdata(xinit)
        self.pl_fit.set_ydata(yinit)
        self.update_plot()

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
                self.but_add_peak()
                # Get last peak
                current_prefix = [mod[0] for mod in self.models][-1]

                # Set model parameters
                xdata, ydata, yerror = self.gen_data()
                sigma = (np.max(xdata) - np.min(xdata)) / 5
                amp = event.ydata * sigma * 3
                # Center
                self.peakvars[current_prefix][1][1].set(event.xdata)  # [Centre][value]
                self.peakvars[current_prefix][1][2].set(np.min(xdata))  # [Centre][min]
                self.peakvars[current_prefix][1][3].set(np.max(xdata))  # [Centre][max]
                # Sigma
                self.peakvars[current_prefix][2][1].set(sigma)  # [Sigma][value]
                self.peakvars[current_prefix][2][2].set(xdata[1]-xdata[0])  # [Sigma][min]
                self.peakvars[current_prefix][2][3].set(3 * sigma)  # [Sigma][max]
                # Amplitude
                self.peakvars[current_prefix][0][1].set(amp)  # [Amplitude][value]
                self.peakvars[current_prefix][0][2].set(0)  # [Amplitude][min]
                self.peakvars[current_prefix][0][3].set(np.inf)  # [Amplitude][max]

                # Update model
                mod, pars = self.gen_model()

                # Update plot
                xinit = np.linspace(xdata.min(), xdata.max(), xdata.size * 10)
                yinit = mod.eval(params=pars, x=xinit)
                self.pl_fit.set_xdata(xinit)
                self.pl_fit.set_ydata(yinit)
                self.update_plot()
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

    def but_add_peak(self):
        """Button add peak"""
        current_prefix = [mod[0] for mod in self.models]
        if 'bkg_' not in current_prefix:
            bkg_name = self.bkg_model.get()
            self.add_model(bkg_name)
        model_name = self.peak_model.get()
        self.add_model(model_name)

    def but_update(self):
        """Button update plot"""
        xdata, ydata, yerror = self.gen_data()
        if self.mask is not None and len(self.mask) == len(xdata):
            xdata = xdata[~self.mask]
            ydata = ydata[~self.mask]
            yerror = yerror[~self.mask]
        # Get model
        model, pars = self.gen_model()
        # Update plot
        self.plot_data()
        xinit = np.linspace(xdata.min(), xdata.max(), xdata.size * 10)
        yinit = model.eval(params=pars, x=xinit)
        self.pl_fit.set_xdata(xinit)
        self.pl_fit.set_ydata(yinit)
        self.update_plot()

    def but_clear_peak(self):
        """Button clear"""
        self.remove_models()

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

    def f_exit(self):
        self.root.destroy()


if __name__ == '__main__':
    # Run GUI
    gui = FittingGUI()

