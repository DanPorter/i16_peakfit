"""
i16_peakfit.tknexus_selector

GUI to select x, y, error fields in nexus data
"""

import os
import numpy as np
import tkinter as tk
from i16_peakfit import tk_widgets
from i16_peakfit.tk_widgets import (TF, BF, SF, HF, MF, LF,
                                    bkg, ety, btn, btn2, opt, btn_active, opt_active, txtcol,
                                    ety_txt, btn_txt, opt_txt, ttl_txt)
from i16_peakfit import nexus_loader


def get_filenames():
    """filedialog widget for Nexus Files"""
    filenames = tk_widgets.filedialog.askopenfilenames(
        title='Select Files for batch processing',
        initialdir='.',
        initialfile='data.nxs',
        defaultextension='*.*',
        filetypes=(('Nexus files', "*.nxs"), ('HDF files', "*.hdf5"), ("all files", "*.*"))
    )
    return [f for f in filenames if os.path.isfile(f)]


class NexusSelectorGui:
    """
    A GUI to select fields in a nexus file
        xdata, ydata, edata, values, batch_name = NexusSelectorGui(parent, files).show()
    """

    def __init__(self, parent, nexus_files=None):
        """Initialise"""

        if nexus_files is None:
            self.nexus_files = get_filenames()
        else:
            self.nexus_files = np.asarray(nexus_files, dtype=str).reshape(-1)

        self.array_addresses = nexus_loader.array_addresses(self.nexus_files[0])
        self.all_addresses = nexus_loader.hdf_addresses(self.nexus_files[0])

        # Create Tk inter instance
        self.parent = parent  # i16_peakfit.tkinter_gui.BatchGui()
        self.root = tk.Toplevel(self.parent.root)
        self.root.wm_title('Nexus Data')
        self.root.minsize(width=100, height=300)
        self.root.maxsize(width=1200, height=1200)
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        self.xaddress = tk.StringVar(self.root, self.array_addresses[0])
        self.yaddress = tk.StringVar(self.root, self.array_addresses[-1])
        self.errorfun = tk.StringVar(self.root, 'np.sqrt(np.abs(y)+1)')
        self.batchaddress = tk.StringVar(self.root, 'entry1/scan_command')
        self.batchvalue = tk.StringVar(self.root, '')

        sec = tk.LabelFrame(self.root, text='Nexus Files', relief=tk.RIDGE)
        sec.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=2, pady=2)
        self.txt_f = tk.Text(sec, width=65, wrap=tk.NONE, height=2)
        self.txt_f.insert(tk.INSERT, '\n'.join(self.nexus_files))
        scl = tk.Scrollbar(sec)
        scl.config(command=self.txt_f.yview)
        self.txt_f.config(yscrollcommand=scl.set)
        self.txt_f.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES, padx=0)
        scl.pack(side=tk.LEFT, fill=tk.Y)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        var = tk.Label(frame, text='Nexus Value: ', font=TF)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(frame, textvariable=self.batchaddress, font=TF, width=32, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        var.bind('<Return>', self.get_batchaddress)
        var.bind('<KP_Enter>', self.get_batchaddress)
        var = tk.Button(frame, text='Select', font=BF, command=self.but_batchaddress,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=2)
        # var = tk.OptionMenu(frame, self.batchaddress, *all_addresses, command=self.get_batchaddress)
        # var.config(font=SF, width=24, bg=opt, activebackground=opt_active)
        # var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        # var.pack(side=tk.LEFT)
        var = tk.Label(frame, textvariable=self.batchvalue, width=40, font=TF)
        var.pack(side=tk.LEFT, padx=4)
        self.get_batchaddress()

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        var = tk.Label(frame, text='X: ', font=TF)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(frame, textvariable=self.xaddress, font=TF, width=32, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Button(frame, text='Select', font=BF, command=self.but_xaddress,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=2)
        # var = tk.OptionMenu(frame, self.xaddress, *array_addresses)
        # var.config(font=SF, width=48, bg=opt, activebackground=opt_active)
        # var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        # var.pack(side=tk.LEFT)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        var = tk.Label(frame, text='Y: ', font=TF)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(frame, textvariable=self.yaddress, font=TF, width=32, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Button(frame, text='Select', font=BF, command=self.but_yaddress,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT, padx=2)
        # var = tk.OptionMenu(frame, self.yaddress, *array_addresses)
        # var.config(font=SF, width=48, bg=opt, activebackground=opt_active)
        # var["menu"].config(bg=opt, bd=0, activebackground=opt_active)
        # var.pack(side=tk.LEFT)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        var = tk.Label(frame, text='Error: ', font=TF)
        var.pack(side=tk.LEFT, padx=2)
        var = tk.Entry(frame, textvariable=self.errorfun, font=TF, width=32, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT, padx=2)

        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        var = tk.Button(frame, text='Launch', font=BF, command=self.f_exit,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=2)

        "-------------------------Start Mainloop------------------------------"
        #self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
        #self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def get_batchaddress(self, event=None):
        """Get dataset value of hdf address"""
        batchaddress = self.batchaddress.get()
        values = nexus_loader.load_hdf_values(self.nexus_files[0], batchaddress, 'None')
        self.batchvalue.set(str(values[0]))

    def but_batchaddress(self):
        """Select address"""
        out = tk_widgets.SelectionBox(self.root, self.all_addresses, title='Nexus Value', multiselect=False).show()
        if out:
            self.batchaddress.set(out[0])
            self.get_batchaddress()

    def but_xaddress(self):
        """Select address"""
        out = tk_widgets.SelectionBox(self.root, self.array_addresses, title='X dataset', multiselect=False).show()
        if out:
            self.xaddress.set(out[0])

    def but_yaddress(self):
        """Select address"""
        out = tk_widgets.SelectionBox(self.root, self.array_addresses, title='Y dataset', multiselect=False).show()
        if out:
            self.yaddress.set(out[0])

    def get_addresses(self):
        xaddress = self.xaddress.get()
        yaddress = self.yaddress.get()
        errorfun = self.errorfun.get()
        batchaddress = self.batchaddress.get()
        return xaddress, yaddress, errorfun, batchaddress

    def get_data(self):
        xaddress, yaddress, errorfun, batchaddress = self.get_addresses()
        xdata = nexus_loader.load_hdf_array(self.nexus_files, xaddress)
        ydata = nexus_loader.load_hdf_array(self.nexus_files, yaddress)
        edata = [eval(errorfun, None, {'x': x, 'y': y}) for x, y in zip(xdata, ydata)]
        values = nexus_loader.load_hdf_values(self.nexus_files, batchaddress)
        return xdata, ydata, edata, values

    def f_exit(self):
        self.root.destroy()

    def show(self):
        """return xdata, ydata, edata, values, batch_name on exit"""
        self.root.deiconify()
        self.root.wait_window()
        batch_name = self.batchaddress.get()
        xdata, ydata, edata, values = self.get_data()
        return xdata, ydata, edata, values, batch_name

