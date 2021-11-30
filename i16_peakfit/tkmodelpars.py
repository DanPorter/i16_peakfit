"""
tkinter gui
"""

import tkinter as tk

from i16_peakfit import fitting
from i16_peakfit.tk_widgets import (TF, BF, SF, HF, MF, LF,
                                    bkg, ety, btn, btn2, opt, btn_active, opt_active, txtcol,
                                    ety_txt, btn_txt, opt_txt, ttl_txt)
from i16_peakfit.functions import shortstr


class ModelParsGUI:
    """
    A standalone GUI for Fitting
    """

    def __init__(self, fittinggui, model, pars=None, title=None):
        """Initialise"""

        if pars is None:
            pars = model.make_params()
        self.pars = pars

        if hasattr(model, 'components'):
            self.components = model.components
        else:
            self.components = [model]

        if title is None:
            title = 'Model Parameters'

        # Create Tk inter instance
        self.parent = fittinggui  # i16_peakfit.tkinter_gui.FittingGui()
        self.root = tk.Toplevel(self.parent.root)
        self.root.wm_title(title)
        self.root.minsize(width=100, height=300)
        self.root.maxsize(width=1200, height=1200)
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        self.peak_model = tk.StringVar(self.root, 'Gaussian')
        self.bkg_model = tk.StringVar(self.root, 'Linear')

        frm = tk.Frame(self.root)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        models = [mod.capitalize() for mod in fitting.PEAK_MODELS if mod in fitting.MODELS]
        var = tk.OptionMenu(frm, self.peak_model, *models)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        models = [mod.capitalize() for mod in fitting.BACKGROUND_MODELS if mod in fitting.MODELS]
        var = tk.OptionMenu(frm, self.bkg_model, *models)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Add Peak Model', font=BF, command=self.but_add_peak,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Button(frm, text='Replace Background Model', font=BF, command=self.but_replace_bkg,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Button(frm, text='Clear all', font=BF, command=self.but_clear,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=2)

        "------------------------ Scollable Canvas ----------------------------------"
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        can = tk.Canvas(frame, height=800, width=1000)
        self.canvas = can
        scl = tk.Scrollbar(frame, orient=tk.VERTICAL, command=can.yview)

        self.peaksec = tk.Frame(can)
        self.peaksec.bind(
            "<Configure>",
            lambda e: can.configure(
                scrollregion=can.bbox("all")
            )
        )
        can.create_window((0, 0), window=self.peaksec, anchor="nw")
        can.configure(yscrollcommand=scl.set)
        # can.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        can.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scl.pack(side=tk.RIGHT, fill=tk.Y)

        "------------------------ Add Models ----------------------------------------"
        self.tkvars = {}  # tkvars[prefix][par][value, min, max, vary]
        for mod in self.components:
            self.create_model(mod)

        "------------------------ Add Buttons ---------------------------------------"
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        var = tk.Button(frame, text='Update & Exit', font=BF, command=self.but_updateexit,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=2)

        "-------------------------Start Mainloop------------------------------"
        self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        self.root.mainloop()

    "------------------------------------------------------------------------"
    "------------------------------UI Functions------------------------------"
    "------------------------------------------------------------------------"

    def create_model(self, component):
        # Find Model type
        mod_name = component.func.__name__.capitalize()
        prefix = component.prefix

        if prefix in self.tkvars:
            self.remove_model(prefix)
        self.tkvars[prefix] = {}

        # Build frame
        sec = tk.LabelFrame(self.peaksec, text=prefix, relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, padx=1, pady=1)
        # sec.bind("<MouseWheel>", self._on_mousewheel)
        self.tkvars[prefix]['frame'] = sec

        if 'bkg' in prefix:
            models = [m.capitalize() for m in fitting.BACKGROUND_MODELS if m in fitting.MODELS]
        else:
            models = [m.capitalize() for m in fitting.PEAK_MODELS if m in fitting.MODELS]
        self.tkvars[prefix]['mod_name'] = tk.StringVar(self.peaksec, mod_name)
        mod_update = self.replace_model_factory(prefix)

        frm = tk.Frame(sec)
        frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=1)
        # frm.bind("<MouseWheel>", self._on_mousewheel)
        var = tk.OptionMenu(frm, self.tkvars[prefix]['mod_name'], *models, command=mod_update)
        var.config(font=SF, width=12, bg=opt, activebackground=opt_active)
        var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(frm, text='Remove', font=BF, command=lambda event=None: self.remove_model(prefix),
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.RIGHT, padx=2)

        model_pars = [pname for pname in self.pars if prefix in pname]
        self.create_pars(sec, prefix, model_pars)

    def create_pars(self, tkframe, model_prefix, model_pars):

        sec = tk.Frame(tkframe)
        sec.pack(side=tk.TOP, fill=tk.X, expand=tk.YES, pady=1)
        # sec.bind("<MouseWheel>", self._on_mousewheel)
        self.tkvars[model_prefix]['pars_frame'] = sec
        self.tkvars[model_prefix]['model_pars'] = model_pars

        for par in model_pars:
            pval = tk.StringVar(tkframe, '%.5g' % self.pars[par].value)
            pmin = tk.StringVar(tkframe, '%.5g' % self.pars[par].min)
            pmax = tk.StringVar(tkframe, '%.5g' % self.pars[par].max)
            pref = tk.IntVar(tkframe, int(self.pars[par].vary))
            expr = self.pars[par].expr
            self.tkvars[model_prefix][par] = {
                'value': pval,
                'min': pmin,
                'max': pmax,
                'vary': pref,
                'expr': expr,
            }
            update = self.update_factory(model_prefix, par, pval)

            frm = tk.Frame(sec)
            frm.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
            # frm.bind("<MouseWheel>", self._on_mousewheel)

            if expr is not None:
                var = tk.Label(frm, text='%s' % par, width=12, font=SF)
                var.pack(side=tk.LEFT, padx=1)
                var = tk.Label(frm, textvariable=pval, width=12, font=SF)
                var.pack(side=tk.LEFT, padx=1)
                var = tk.Label(frm, text='expr=%s' % shortstr(expr), font=SF)
                var.pack(side=tk.LEFT, padx=1)
            else:
                var = tk.Label(frm, text='%s' % par, width=12, font=SF)
                var.pack(side=tk.LEFT, padx=1)
                var = tk.Label(frm, text='Value:', font=TF)
                var.pack(side=tk.LEFT, padx=2)
                var = tk.Entry(frm, textvariable=pval, font=TF, width=10, bg=ety, fg=ety_txt)
                var.pack(side=tk.LEFT, padx=2)
                var.bind('<Return>', update)
                var.bind('<KP_Enter>', update)
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

    def replace_model_factory(self, update_prefix):
        def f(event):
            new_model_name = self.tkvars[update_prefix]['mod_name'].get()
            new_model = fitting.getmodel(new_model_name)(prefix=update_prefix)
            new_pars = new_model.make_params()
            prefix_parnames = [pname for pname in self.pars if update_prefix in pname]
            # remove any model pars from global list that aren't in new model (e.g. pvoight>gauss gamma)
            for name in prefix_parnames:
                if name not in new_pars:
                    del self.pars[name]
            # add any model pars to global list that are in new model (e.g. gauss>pvoight gamma)
            for name in new_pars:
                if name not in prefix_parnames:
                    self.pars[name] = new_pars[name]  # this doesn't work because new_pars expr has wrong names

            # Replace ui elements
            prefix_parnames = [pname for pname in self.pars if update_prefix in pname]
            self.tkvars[update_prefix]['pars_frame'].destroy()
            for name in prefix_parnames:
                if name in self.tkvars[update_prefix]:
                    del self.tkvars[update_prefix][name]
            self.create_pars(self.tkvars[update_prefix]['frame'], update_prefix, prefix_parnames)

        return f

    def update_factory(self, update_prefix, update_parname, update_pval):
        def f(event):
            self.pars[update_parname].set(value=float(update_pval.get()))
            self.update_all(update_prefix)

        return f

    def update_all(self, update_prefix):
        for parname in self.tkvars[update_prefix]['model_pars']:
            value = self.pars[parname].value
            self.tkvars[update_prefix][parname]['value'].set('%.5g' % value)
        self.parent_update()

    def update_pars(self, new_pars):
        self.pars.update(new_pars)
        for parname in new_pars:
            value = new_pars[parname].value
            for update_prefix in self.tkvars:
                if parname in self.tkvars[update_prefix]:
                    self.tkvars[update_prefix][parname]['value'].set('%.5g' % value)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 60)), "units")

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def remove_model(self, model_prefix):
        self.tkvars[model_prefix]['frame'].destroy()
        del self.tkvars[model_prefix]

    def remove_models(self):
        """Remove all models"""
        for model_prefix in self.tkvars:
            self.remove_model(model_prefix)

    def gen_model(self):
        """Create model, pars from input boxes"""
        # Get model
        model_list = []
        par_list = []
        for prefix in self.tkvars:
            mod_name = self.tkvars[prefix]['mod_name'].get()
            model = fitting.getmodel(mod_name)(prefix=prefix)
            mod_pars = model.make_params()
            for parname in mod_pars:
                pval = float(self.tkvars[prefix][parname]['value'].get())
                pmin = float(self.tkvars[prefix][parname]['min'].get())
                pmax = float(self.tkvars[prefix][parname]['max'].get())
                vary = bool(self.tkvars[prefix][parname]['vary'].get())
                expr = self.tkvars[prefix][parname]['expr']
                if expr is None:
                    mod_pars[parname].set(value=pval, min=pmin, max=pmax, vary=vary)
            model_list += [model]
            par_list += [mod_pars]

        if len(model_list) == 0:
            return None, {}
        full_model = model_list[0]
        all_pars = par_list[0]
        for model, mod_pars in zip(model_list[1:], par_list[1:]):
            full_model += model
            all_pars.update(mod_pars)
        return full_model, all_pars

    def parent_update(self):
        """Update parent gui"""
        model, pars = self.gen_model()
        self.parent.update_model(model, pars)

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def but_updateexit(self):
        """Update button"""
        self.parent_update()
        self.f_exit()

    def but_add_peak(self):
        mod_name = self.peak_model.get()
        n = 1
        while 'p%d_' % n in self.tkvars:
            n += 1
        new_prefix = 'p%d_' % n

        new_model = fitting.getmodel(mod_name)(prefix=new_prefix)
        new_pars = new_model.make_params()
        self.pars.update(new_pars)
        self.create_model(new_model)

    def but_replace_bkg(self):
        mod_name = self.bkg_model.get()
        if 'bkg_' in self.tkvars:
            self.remove_model('bkg_')

        new_model = fitting.getmodel(mod_name)(prefix='bkg_')
        new_pars = new_model.make_params()
        self.pars.update(new_pars)
        self.create_model(new_model)

    def but_clear(self):
        self.remove_models()

    def f_exit(self):
        self.root.unbind_all("<MouseWheel>")
        self.root.destroy()


