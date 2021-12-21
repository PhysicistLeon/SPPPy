# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:39:02 2021.

@author: THzLab
"""

# Импортируем все из библиотеки TKinter
from tkinter import *
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.lib import scimath as SM

import matplotlib.pyplot as plt
import pandas as pd
from SPPPy import *

window = Tk()


# --------------------------------------------------------------------------
# -------------------- Classes ---------------------------------------------
# --------------------------------------------------------------------------


class VerticalScrolledFrame(Frame):
    """ Honestly stealed from internet.
    A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling
    """
    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)

        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)


class Parametr:
    """
    Parametr type.

    Contains name, Scale, Entyr and dim_label optional
    """

    def __init__(self, my_frame, name, min_val=1, max_val=10, step=1, have_dim=False, mypar_scl=None, mypar_txt = None):
        """
        Init.
        Create all forms

        Parameters
        ----------
        my_frame : Frame
            External frame to contain all elements.
        name : string
            Name label text.
        min_val : float, optional
            minimum scale value. The default is 1.
        max_val : float, optional
            miximum scale value. The default is 10.
        step : float, optional
            Scale step. The default is 1.
        have_dim : boolean, optional
            Define if dim_label shows. The default is False.
        """
        self.my_frame = my_frame
        self.minimum = min_val
        self.maximum = max_val
        self.havedim = have_dim

        self.frm_x = Frame(my_frame, width=20, height=500)
        self.frm_x.pack(fill=X)

        self.lbl_x = Label(self.frm_x, text=f"{name} = ", font=("Calibri", 9))
        self.lbl_x.pack(side=LEFT)

        self.scl_x = Scale(self.frm_x, orient=HORIZONTAL, showvalue=False,
            sliderlength=10, from_=self.minimum, to=self.maximum,
            resolution=step, command=self.scl_x_change, variable = mypar_scl)
        self.scl_x.pack(side=LEFT)

        # Look if text labels is connected
        self.txt_x = StringVar()
        if mypar_txt is None:
            self.txt_x = StringVar()
        else:
            self.txt_x = mypar_txt

        self.txt_x.trace_add('write', self.ent_x_change)
        self.ent_x = Entry(self.frm_x, textvariable=self.txt_x, width=6)
        self.ent_x.insert(0, 1)
        self.ent_x.pack(side=LEFT)
        if(self.havedim):
            self.dim_x = Label(self.frm_x, text=spin_dim.get())
            self.dim_x.pack(side=LEFT)

    def ent_x_change(self, *args):
        """
        Move info from Scale to entry.

        Parameters
        ----------
        *args : none.
        """
        if self.ent_x.get() != '' and (self.scl_x.get() != float(self.ent_x.get())):
            my_txt = self.ent_x.get()
            if float(my_txt) > self.maximum:
                my_txt = self.maximum
                self.ent_x.delete(0, END)
                self.ent_x.insert(0, my_txt)
            if float(my_txt) < self.minimum:
                my_txt = self.minimum
                self.ent_x.delete(0, END)
                self.ent_x.insert(0, my_txt)

            self.scl_x.set(my_txt)

            #Autodraw
            for i in range(0,len(R_tabs)):
                if R_tabs[i].chk_state.get():
                    R_tabs[i].build_curve()

    def scl_x_change(self, *args):
        """
        Move info from Entry to Scale.

        Parameters
        ----------
        *args : none.
        """
        if (float(self.scl_x.get()) != float(self.ent_x.get())):
            self.ent_x.delete(0, END)
            self.ent_x.insert(0, self.scl_x.get())

            #Autodraw
            for i in range(0,len(R_tabs)):
                if R_tabs[i].chk_state.get():
                    R_tabs[i].build_curve()

    def dim_set(self, new_dim):
        """
        Set dim_label.

        Parameters
        ----------
        new_dim : string
            new dim_label text.
        """
        if self.havedim:
            self.dim_x['text'] = new_dim

    def get(self):
        """
        Get parametr value.

        Returns
        -------
        a : float
            value from scale with dim_val, if it defineв.
        """
        if self.havedim:
            a = self.scl_x.get() * float(self.dim_x['text'])
        else:
            a = self.scl_x.get()
        return a

    def set_parametr(self, par):
        """
        Set parametr value.

        Parameters
        ----------
        par : int
            Parametr value.
        """
        if self.havedim:
            self.scl_x.set(par/float(self.dim_x['text']))
        else:
            self.scl_x.set(par)


# -------------------- Layers classes --------------------------------------


class Layer():
    """
    Layer type.

    Can be metal or dielectric
    """

    # Global parametrs
    is_last = True  # if layer is last needs to hide d
    is_set = False  # hide or show layer of this type, when type swithces

    def __init__(self, parent_frame, my_master):
        """
        Init for class.

        Parameters
        ----------
        parent_frame : Frame
            External frame to contain layer.
        my_master : Slass: layers container
            Link to god to prey for reincarnation or death.

        """
        self.parent_frame = parent_frame
        self.my_master = my_master
        self.my_frame = Frame(parent_frame)  # frame to hide and show layer

        # side frame with controls
        self.control_frame = Frame(self.my_frame, width=5)
        self.control_frame.pack(side=RIGHT, fill=Y)
        self.del_layer = Button(self.control_frame, text="x",
                                font=("Times new roman", 9),
                                command=lambda: self.my_master.kill(self))
        self.del_layer.pack(side=TOP)

        # set main parametrs
        self.set_type()
        self.d = Parametr(self.my_frame, ' d', 1, 100, 0.1, True)
        if self.is_last:
            self.d.frm_x.pack_forget()

    def grad_disable(self):
        """
        Null proc to redefine in heir class.

        Returns none.
        """
        return

    def grad_enable(self):
        """
        Null proc to redefine in heir class.

        Returns none.
        """
        return

    def set_type(self):
        """
        Null proc to redefine in heir class.

        Returns none.
        """
        return

    def turn_on(self):
        """
        Show thus layer type when swithced.

        Returns none.
        """
        if not self.is_set:
            self.is_set = True
            self.my_frame.pack(fill=BOTH, expand=True)

    def turn_off(self):
        """
        Hide thus layer type when swithced.

        Returns
        -------
        None.
        """
        if  self.is_set:
            self.is_set = False
            self.my_frame.pack_forget()

    def dim_change(self, val):
        """
        Change dim_label.

        Parameters
        ----------
        val : string
            new dim_label text.
        """
        self.d.dim_set(val)

    def set_last(self):
        """
        Hide d if layer is last (semi infinite layer).

        Returns
        -------
        None.
        """
        if not self.is_last:
            self.is_last = True
            self.d.frm_x.pack_forget()

    def unset_last(self):
        """
        Show d if layer is not last (thin layer).

        Returns none.
        """
        if self.is_last:
            self.is_last = False
            self.d.frm_x.pack(fill=X)


class Layer_m(Layer):
    """
    Metall layer.

    Layer heir
    """

    def set_type(self):
        """
        Overloading - set metall parametrs.

        Returns none
        """
        self.change_type_d = Button(self.control_frame, text="d",
                                font=("Times new roman", 9),
                                command=lambda: self.my_master.switch(self, 'd'))
        self.change_type_d.pack(side=BOTTOM)

        self.change_type_g = Button(self.control_frame, text="g",
                                font=("Times new roman", 9),
                                command=lambda: self.my_master.switch(self, 'g'))
        self.change_type_g.pack(side=BOTTOM)

        self.eps1 = Parametr(self.my_frame, ' ε\'', -10000, -1)
        self.eps2 = Parametr(self.my_frame, 'ε\'\'', 0.1, 5000, 0.1)

    def grad_disable(self):
        """
        Disables grad button, ig grad layer shown.

        Returns none.
        """
        self.change_type_g['state'] = DISABLED

    def grad_enable(self):
        """
        Enables grad button, ig grad layer shown.

        Returns none.
        """
        self.change_type_g['state'] = NORMAL

    def get_parametrs(self):
        """ Take actual layer parametrs.

        Returns
        -------
        complex
            Complex metall permittivity.
        float
            Layer thickness.
        """
        return (self.eps1.get() + 1j*self.eps2.get()), self.d.get()

    def set_parametrs(self, eps_d):
        """ Set layer parametrs.

        Parameters
        ----------
        eps_d : array
            [imag_eps (as its a metal), real_d].
        """
        self.eps1.set_parametr(re(eps_d[0]))
        self.eps2.set_parametr(im(eps_d[0]))
        self.d.set_parametr(re(eps_d[1]))


class Layer_d(Layer):
    """ Dielectric layer.

    Layer heir
    """

    def set_type(self):
        """ Overloading - set dielectric parametrs.

        Returns none
        """
        self.turn_on()
        self.change_type_m = Button(self.control_frame, text="m",
                                font=("Times new roman", 9),
                                command=lambda: self.my_master.switch(self, 'm'))
        self.change_type_m.pack(side=BOTTOM)    

        self.change_type_g = Button(self.control_frame, text="g",
                                font=("Times new roman", 9),
                                command=lambda: self.my_master.switch(self, 'g'))
        self.change_type_g.pack(side=BOTTOM)

        self.eps = Parametr(self.my_frame, ' ε\'', 1, 10)

    def grad_disable(self):
        """ Disables grad button, ig grad layer shown.

        Returns none.
        """
        self.change_type_g['state'] = DISABLED

    def grad_enable(self):
        """ Enables grad button, ig grad layer shown.

        Returns none.
        """
        self.change_type_g['state'] = NORMAL

    def get_parametrs(self):
        """ Take actual layer parametrs.

        Returns
        -------
        float
            Dielectric permittivity.
        float
            Layer thickness.
        """
        return (self.eps.get(), self.d.get())

    def set_parametrs(self, eps_d):
        """ Set layer parametrs.

        Parameters
        ----------
        eps_d : array
            [real_eps (as its a dielectric), real_d].
        """
        self.eps.set_parametr(re(eps_d[0]))
        self.d.set_parametr(re(eps_d[1]))


class Layer_grad(Layer):
    """ Gradient layer.

    Layer heir
    """

    def set_type(self):
        """ Overloading - set dielectric parametrs.

        Returns none
        """
        self.change_type_m = Button(self.control_frame, text='m',
                                font=("Times new roman", 9),
                                command=lambda: self.my_master.switch(self, 'm'))
        self.change_type_m.pack(side=BOTTOM)  

        self.change_type_d = Button(self.control_frame, text='d',
                                font=("Times new roman", 9),
                                command=lambda: self.my_master.switch(self, 'd'))
        self.change_type_d.pack(side=BOTTOM)  

        # Grad parametrs:
        self.grad_frame = Frame(self.my_frame)
        self.grad_frame.pack(fill=X)

        self.grad_func_var = StringVar()
        self.grad_func_values = ("a*x+c", "a*(x+b)**2",
                                 "a*(x+b)**3")

        self.spin_grad_func = ttk.Combobox(self.grad_frame, values=self.grad_func_values,
                                           state="readonly")
        self.spin_grad_func.current(0)
        self.spin_grad_func.bind("<<ComboboxSelected>>")
        self.spin_grad_func.pack(side=LEFT)

        self.A = Parametr(self.my_frame, 'A', step=0.1)
        self.B = Parametr(self.my_frame, 'B', step=0.1)

    def get_parametrs(self):
        """ Take actual layer parametrs.

        Returns
        -------
        float
            Dielectric permittivity.
        float
            Layer thickness.
        """
        vals = (self.spin_grad_func.get(), self.A.get(), self.B.get(), self.d.get())
        return vals

    def set_parametrs(self, arr):
        """ Set layer parametrs.

        Parameters
        ----------
        arr : array
            [function, A, B, d].
        """
        self.spin_grad_func.current(self.grad_func_values.index(arr[0]))
        self.A.set_parametr(float(arr[1]))
        self.B.set_parametr(float(arr[2]))
        self.d.set_parametr(arr[3])


class Layers_container:
    """ Container of all layers.

    Fit to canvas of scrollable frame
    """
     # Global gradient function parametrs to rescue
    grad_par_rescue = ["a*x+c", 1, 1, 1]

    n_frames = []  # Frame for variants of a layer
    n_layer = []  # Layers itself

    def __init__(self, parent_frame):
        """ Parameters.

        ----------
        parent_frame : Frame
            External frame, canvas of scrollable frame.
        """
        # A = grad_par_rescue[0]
        # B = grad_par_rescue[1]
        self.grad_profiles = {
            "a*x+c":        lambda x: (self.grad_par_rescue[1]/2 - 2.5) * x + 1.5 + self.grad_par_rescue[2]/2,
            "a*(x+b)**2":   lambda x: self.grad_par_rescue[1]* 2 * (x-0.5)**2 + self.grad_par_rescue[2]/2 + 0.5,
            "a*(x+b)**3":   lambda x: self.grad_par_rescue[1]* 2 * (x-0.5)**3 + self.grad_par_rescue[2]/2}
        self.parent_frame = parent_frame

    def add_layer(self):
        """ Add frame and all types of layer.

        Returns none.
        """

        # Set ex-last as not last
        if len(self.n_layer) > 0:
            for ii in range(0, len(self.n_layer[0])):
                self.n_layer[-1][ii].unset_last()


        self.n_frames.append(Frame(self.parent_frame, width=20, height=500, relief=RAISED, bd=2))
        self.n_frames[-1].pack(fill=X)
        self.n_layer.append((Layer_d(self.n_frames[-1], self),
                             Layer_m(self.n_frames[-1], self),
                             Layer_grad(self.n_frames[-1], self)))
        
        # grad button turn off
        for j in range(0, len(self.n_frames)):
            if self.n_layer[j][2].is_set:
                for ii in range(0, len(self.n_layer[0])):
                    self.n_layer[-1][ii].grad_disable()
                break

    def switch(self, exiled=None, invoked='d'):
        """ Switch layer types in the frame.

        Parameters
        ----------
        exiled : Layer
            Who ask to be exiled.
        invoked : Char
            Layer type to be invoked.
        """
        if  exiled is None:
            exiled = self.n_layer[-1][0]

        # Layer type to evoke
        if invoked == 'd':
            wake = 0
        elif invoked == 'm':
            wake = 1
        else:
            wake = 2 # then gradient

        # Search active layer number
        for lay_num in range(0, len(self.n_frames)):
            if self.n_layer[lay_num][0] == exiled:
                sleep = 0
                break
            if self.n_layer[lay_num][1] == exiled:
                sleep = 1
                break
            if self.n_layer[lay_num][2] == exiled:
                sleep = 2
                break

        # Gradient layer involved: Turn off all grad buttons
        if wake == 2:
            # Load parametrs
            self.grad_par_rescue[-1] = self.grad_par_rescue[-1] * float(spin_dim.get())
            self.n_layer[lay_num][wake].set_parametrs(self.grad_par_rescue)
            # Turn off al
            for i in range(0, len(self.n_frames)):
                for j in range(0, len(self.n_layer[0])):
                    self.n_layer[i][j].grad_disable()

        # Turn on all grad but
        if sleep == 2:
            self.save_grad()

            for i in range(0, len(self.n_frames)):
                for j in range(0, len(self.n_layer[0])):
                    self.n_layer[i][j].grad_enable()

        self.n_layer[lay_num][sleep].turn_off()
        self.n_layer[lay_num][wake].turn_on()

    def kill(self, unit_to_kill):
        """ Delete layer and it's frame.

        Parameters
        ----------
        unit_to_kill :  Layer
            Who prey for death.
        """
        # If its the last then stop!
        if len(self.n_layer) == 1:
            messagebox.showerror("Error", "System must have at least two layers!")
            return

        # search layer and save number to i
        for i in range(0, len(self.n_frames)):
            if unit_to_kill == self.n_layer[i][0] or unit_to_kill == self.n_layer[i][1] or unit_to_kill == self.n_layer[i][2]:
                break
        
        # If killed is grad so free all grad switch buttons
        if unit_to_kill == self.n_layer[i][2]: # free all grad buttons
            for j in range(0, len(self.n_frames)):
                for k in range(0, len(self.n_layer[0])):
                    self.n_layer[j][k].grad_enable()
        
        self.n_layer[i][0].parent_frame.destroy()

        # Remove talking layers from array
        self.n_layer.pop(i)
        self.n_frames.pop(i)

        # If buried frame was the last, appointing previous as the last
        if len(self.n_layer) > 0:
            for k in range(0, len(self.n_layer[0])):
                self.n_layer[-1][k].set_last()

    def dim_change(self, val):
        """ Change dimension of layer's parametrs.

        Parameters
        ----------
        val : string
            New dimension.
        """
        for layr in self.n_layer:
            for k in range(0, len(self.n_layer[0])):
                layr[k].dim_change(val)

    def save_grad(self):
        """ Find gradient laeyrs and pull parametrs for rescue.
        
        Returns none.
        """
        for i in range(0, len(self.n_frames)):
            if self.n_layer[i][2].is_set:
                rescue = self.n_layer[i][2].get_parametrs()
                for i in range(0, len(rescue) - 1):
                    self.grad_par_rescue[i] = rescue[i]
                    self.grad_par_rescue[len(rescue) - 1] = rescue[len(rescue) - 1] / float(spin_dim.get())
                    
    def get_grad(self):
        """ Get gradient layer parametrs.

        Returns
        -------
        [int, lambda]
            [grad layer number, lambda function in (0, 1)].
        """
        self.save_grad()
        grad_num = -1
        for i in range(0, len(self.n_frames)):
            if self.n_layer[i][2].is_set:
                grad_num = i + 1
        return grad_num, self.grad_profiles[self.grad_par_rescue[0]]

    def get_system_parametrs(self):
        """ Take dielectric permittivityes and thicknesses of layers.

        Returns
        -------
        eps_out : [float]
            dielectric permittivityes.
        d_out : [float]
            thicknesses of layers, 0 for first and last as semi infinite.
        """
        eps_out = []
        d_out = []
        eps_out.append(eps_prism_par.get())
        d_out.append(0.) # first layer semiinfinite

        for i in range(0, len(self.n_layer)):
            if self.n_layer[i][0].is_set:
                c = self.n_layer[i][0].get_parametrs()
            elif self.n_layer[i][1].is_set:
                c = self.n_layer[i][1].get_parametrs()
            else: 
                c = (0, self.n_layer[i][2].d.get()) # means that layer is gradient
            eps_out.append(c[0])
            d_out.append(c[1])

        d_out[-1] = 0 #last layer semiinfinite

        self.save_grad()

        return eps_out, d_out

    def set_system_parametrs(self, eps_d):
        """ Set parametrs of homogeneous layers (not for prysm).

        Parameters
        ----------
        eps_d : [complex, float]
            array of eps and d for each layer.
        """
        # clean all layers

        while len(self.n_layer) > 0:
            self.n_layer[0][0].parent_frame.destroy()
            self.n_layer.pop(0)
            self.n_frames.pop(0)
        
        # add new layers and fill parametrs
        for i in range(0, eps_d.shape[0]):
            self.add_layer()
            if im(eps_d[i,0]) != 0: # metal
                self.n_layer[i][1].set_parametrs([eps_d[i,0], re(eps_d[i,1])])
                self.switch(self.n_layer[i][0], 'm')
            elif abs(eps_d[i,0]) != 0:
                self.n_layer[i][0].set_parametrs([re(eps_d[i,0]), re(eps_d[i,1])])
                self.switch(self.n_layer[i][0], 'd')
            else: self.switch(self.n_layer[i][0], 'g')



# -------------------- Drawing tabs classes ---------------------------------


class DrawingType_tab:
    """ Common class for all types of drawings.

    Draw in tabs
    """

    def __init__(self, parent_frame, my_par_lam=None, my_par_scl=None, my_par_theta=None, my_par_theta_scl=None):
        """ Init common logic frames

        Parameters
        ----------
        parent_frame : Frame
            External frame.
        """
        self.my_par_lam = my_par_lam
        self.my_par_scl = my_par_scl

        # for theta defined tabs
        self.my_par_theta = my_par_theta
        self.my_par_theta_scl = my_par_theta_scl
        self.lambda_range = np.linspace(1e-9, 15e-5, 500)

        self.parent_frame = parent_frame
        # self.parent_tab = parent_tab
        self.frame_draw_bt = Frame(parent_frame, bd=5)
        self.frame_draw_bt.pack(side=BOTTOM, fill=X)
        self.draw_my_parametr()
        self.draw_btn = Button(self.frame_draw_bt,
                               text="= > Draw dispersion curve < =",
                               font=("Times new roman", 10),
                               command=lambda: self.build_curve())
        self.draw_btn.pack(side=LEFT)
        self.chk_state = BooleanVar()
        self.chk_state.set(False)
        self.chk = Checkbutton(self.frame_draw_bt,
            text='Autodraw with parametrs change', var=self.chk_state)
        self.chk.pack(side=RIGHT)

        # Draw window
        self.canvas_frame = Frame(self.parent_frame, relief=SUNKEN, bd=3)
        self.canvas_frame.pack(fill=BOTH, expand=True)
        self.canvas_frame_int = Frame(self.canvas_frame, bd=2)
        self.canvas_frame_int.pack(fill=BOTH, expand=True)

        self.reflection_curve = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.reflection_curve,
                                        self.canvas_frame_int)
        self.canvas.get_tk_widget().place(relx=-0.02, rely=-0.1,
                                          relwidth=1.1, relheight=1.15)
        self.R_plot = self.reflection_curve.add_subplot(1, 1, 1)
        self.mark_axex()

    def plot_curve(self, x, y):
        """ Procedure to draw on canvas.

        Parameters
        ----------
        x : [float]
            Abscissas array.
        y : [float]
            Ordinates array.
        """
        self.R_plot.clear()
        self.mark_axex()
        self.R_plot.plot(x, y)
        self.canvas.draw()

    def draw_my_parametr(self):
        """ Proc to reload.

        Returns none.
        """
        return

    def mark_axex(self):
        """ Proc to reload.

        Returns none.
        """
        return

    def build_curve(self):
        """ Proc to reload.

        Returns none.
        """
        return

    def get_parametr(self):
        """ Get tab main parametr.

        Returns
        -------
        float
            wavelength un unit

        """
        return self.lam_par.get()

    def dim_change(self):
        """ Proc to reload.

        Returns none.
        """
        return


class Grad_profile_tab(DrawingType_tab):

    def mark_axex(self):
        """
        Mark plot axes.

        Returns none
        """
        self.R_plot.set_xlabel('d %')
        self.R_plot.set_ylabel('n')

    def build_curve(self):
        """
        Build gradient profile.

        Returns none
        """
        profile = Layers.get_grad()[1]
        n_range = np.linspace(0, 1, 100)
        n_prof = [profile(i) for i in n_range]
        self.plot_curve(n_range, n_prof)

    def dim_change(self):
        """
        Change dimensions.

        Returns none
        """
        return

    def get_parametr(self):
        """
        Get parametrs value.

        Returns none
        """
        return

    def set_parametr(self, par):
        """
        Set parametrs value

        Parameters
        ----------
        par : float
            lambda value
        """
        return


class Re_theta_tab(DrawingType_tab):
    """
    Real R from theta logic frames

    Init parameters
    ----------
    parent_frame : Frame
        External frame.
    """

    def draw_my_parametr(self):
        """
        Re R(theta) parametrs: lambda.

        Returns none.
        """
        self.lam_par = Parametr(self.parent_frame, 'λ', 1,
                                15000, 0.1, have_dim=True, mypar_scl=self.my_par_lam, mypar_txt=self.my_par_scl)

    def mark_axex(self):
        """
        Mark plot axes.

        Returns none
        """
        self.R_plot.set_xlabel('ϴ')
        self.R_plot.set_ylabel('R')

    def build_curve(self):
        """
        Calculate Real R from theta curve.

        Returns none
        """
        a = Layers.get_system_parametrs()
        grad_par = Layers.get_grad()

        Unit = SPP_experiment(w=2.0 * np.pi / self.lam_par.get(),
                              n=SM.sqrt(a[0]), d=a[1], Gradient=grad_par)

        theta_range = np.linspace(0, 90, 1000)
        print(a)
        self.plot_curve(theta_range, Unit.R_theta_re(theta_range))

    def dim_change(self):
        """
        Change dimensions.

        Returns none
        """
        self.lam_par.dim_set(spin_dim.get())
    
    def get_parametr(self):
        """
        Get parametrs value.

        Returns none
        """
        return self.lam_par.get()
    
    def set_parametr(self, par):
        """
        Set parametrs value

        Parameters
        ----------
        par : float
            lambda value
        """
        self.lam_par.set_parametr(par)


class Im_theta_tab(DrawingType_tab):
    """
    Imaginary r from theta logic frames

    Init parameters
    ----------
    parent_frame : Frame
        External frame.
    """
    def draw_my_parametr(self):
        self.lam_par = Parametr(self.parent_frame, 'λ', 1, 15000, 0.1,
                                have_dim=True, mypar_scl=self.my_par_lam, mypar_txt=self.my_par_scl )

    def mark_axex(self):
        """
        Mark plot axes.

        Returns none
        """
        self.R_plot.set_xlabel('Re(r)')
        self.R_plot.set_ylabel('Im(r)')

    def build_curve(self):
        """
        Calculate Imaginary r from theta curve.

        Returns none
        """
        a = Layers.get_system_parametrs()

        Unit = SPP_experiment(w=2.0 * np.pi / self.lam_par.get(),
                          n=SM.sqrt(a[0]), d=a[1])
        tir = Unit.TIR()
        theta_range = np.linspace(tir, 90, 1000)

        b =  Unit.R_theta_cmpl(theta_range)
        x = []
        y = []
        for i in range(0, len(b)):
            x.append(re(b[i]))
            y.append(im(b[i]))
        self.plot_curve(x, y)

    def dim_change(self):
        """
        Change dimensions.

        Returns none
        """
        self.lam_par.dim_set(spin_dim.get())
    
    def get_parametr(self):
        """
        Get parametrs value.

        Returns none
        """
        return self.lam_par.get()
    
    def set_parametr(self, par):
        """
        Set parametrs value

        Parameters
        ----------
        par : float
            lambda value
        """
        self.lam_par.set_parametr(par)


class Re_lambda_tab(DrawingType_tab):
    """
    Real R from theta logic frames

    Init parameters
    ----------
    parent_frame : Frame
        External frame.
    """

    def draw_my_parametr(self):
        """
        Re R(theta) parametrs: lambda.

        Returns none.
        """
        self.theta_par = Parametr(self.parent_frame, 'ϴ', 0,
                                90, 0.1, mypar_scl=self.my_par_theta, mypar_txt=self.my_par_theta_scl)

    def mark_axex(self):
        """
        Mark plot axes.

        Returns none
        """
        self.R_plot.set_xlabel('λ')
        self.R_plot.set_ylabel('R')

    def build_curve(self):
        """
        Calculate Real R from theta curve.

        Returns none
        """
        a = Layers.get_system_parametrs()
        grad_par = Layers.get_grad()

        Unit = SPP_experiment(w=0, n=SM.sqrt(a[0]), d=a[1], Gradient=grad_par)

        self.plot_curve(self.lambda_range, Unit.R_lambda_re(self.theta_par.get(), self.lambda_range))

    def dim_change(self):
        """
        Change dimensions.

        Returns none
        """
        return
    
    def get_parametr(self):
        """
        Get parametrs value.

        Returns none
        """
        return self.theta_par.get()
    
    def set_parametr(self, par):
        """
        Set parametrs value

        Parameters
        ----------
        par : float
            lambda value
        """
        self.theta_par.set_parametr(par)


class Im_lambda_tab(DrawingType_tab):
    """
    Real R from theta logic frames

    Init parameters
    ----------
    parent_frame : Frame
        External frame.
    """

    def draw_my_parametr(self):
        """
        Re R(theta) parametrs: lambda.

        Returns none.
        """
        self.theta_par = Parametr(self.parent_frame, 'ϴ', 0,
                                90, 0.1, mypar_scl=self.my_par_theta, mypar_txt=self.my_par_theta_scl)

    def mark_axex(self):
        """
        Mark plot axes.

        Returns none
        """
        self.R_plot.set_xlabel('Re(r)')
        self.R_plot.set_ylabel('Im(r)')

    def build_curve(self):
        """
        Calculate Real R from theta curve.

        Returns none
        """
        a = Layers.get_system_parametrs()
        grad_par = Layers.get_grad()

        Unit = SPP_experiment(w=0, n=SM.sqrt(a[0]), d=a[1], Gradient=grad_par)

        b =  Unit.R_lambda_cmpl(self.theta_par.get(), self.lambda_range)
        x = []
        y = []
        for i in range(0, len(b)):
            x.append(re(b[i]))
            y.append(im(b[i]))
        self.plot_curve(x, y)

    def dim_change(self):
        """
        Change dimensions.

        Returns none
        """
        return
    
    def get_parametr(self):
        """
        Get parametrs value.

        Returns none
        """
        return self.theta_par.get()
    
    def set_parametr(self, par):
        """
        Set parametrs value

        Parameters
        ----------
        par : float
            lambda value
        """
        self.theta_par.set_parametr(par)

# -------------------- Functions -------------------------------------------


def dim_change(*args):
    """
    Command for Spinbox change.

    *args : None
    """
    Layers.dim_change(spin_dim.get())

    # Autodraw
    for i in range(0, len(R_tabs)):
        R_tabs[i].dim_change()
        if R_tabs[i].chk_state.get():
            R_tabs[i].build_curve()


def save_file():
    """
    Save file proc.

    Returns none.
    """
    print('save file')
    # save layers
    Layers.save_grad()
    a = Layers.get_system_parametrs()

    MyFile = pd.DataFrame(a)
    MyFile = MyFile.T
    layer_count = len(a[0])
    MyFile['type'] = ['layer']*layer_count

    # Get tabs parametrs
    MyFile.loc[layer_count + 0] = [experiment_lambda_scale.get()*float(spin_dim.get()), 0, 'lambda']
    MyFile.loc[layer_count + 1] = [experiment_theta_scale.get(), 0, 'theta']

    # get dimension
    MyFile.loc[layer_count + 2] = [spin_dim.get(), 0, 'dim']
    MyFile.loc[layer_count + 3] = [Layers.grad_par_rescue[0],
                                   Layers.grad_par_rescue[1], 'grad']
    MyFile.loc[layer_count + 4] = [Layers.grad_par_rescue[2],
                                   Layers.grad_par_rescue[3], 'grad']

    # Fill file
    MyFile = MyFile.set_index('type')
    file_name = filedialog.asksaveasfilename(filetypes=[("csv files", "*.csv")])
    if file_name != '':
        MyFile.to_csv(file_name + '.csv', sep=' ')
        print(MyFile)


def open_file():
    """
    Open file proc.

    Returns none.
    """
    print('open file')

    realy = messagebox.askquestion("Open file",
                "Actual set will be erased, are You Sure?", icon='warning')

    if realy == 'yes':
        # Recieving parametrs
        file_name = filedialog.askopenfilename(filetypes=[("csv files", "*.csv")])
        if file_name != '':
            data = pd.read_csv(file_name, sep=' ')
            my_dim = data.values[data['type'] == 'dim'][:, 1][0]

            my_layers = data.values[data['type'] == 'layer'][:, 1:3].astype(complex)

            my_lam = data.values[data['type'] == 'lambda'][:, 1].astype(complex)
            my_thet = data.values[data['type'] == 'theta'][:, 1].astype(complex)

            grad_par = data.values[data['type'] == 'grad'][:, 1:]

            Layers.grad_par_rescue[0] = grad_par[0][0]

            if grad_par[0][1][0]=='(':
                Layers.grad_par_rescue[1] = re(complex(grad_par[0][1][1:-1]))
            else:
                Layers.grad_par_rescue[1] = re(complex(grad_par[0][1]))

            if grad_par[1][0][0]=='(':
                Layers.grad_par_rescue[2] = re(complex(grad_par[1][0][1:-1]))
            else:
                Layers.grad_par_rescue[2] = re(complex(grad_par[1][0]))

            if grad_par[1][1][0]=='(':
                Layers.grad_par_rescue[3] = re(complex(grad_par[1][1][1:-1]))
            else:
                Layers.grad_par_rescue[3] = re(complex(grad_par[1][1]))

            # Set recieved parametrs
            dim_var.set(my_dim)
            dim_change()
            eps_prism_par.set_parametr(re(my_layers[0, 0])) # Prism not in layers class

            Layers.set_system_parametrs(my_layers[1:,:])

            # set lambda
            R_tabs[0].set_parametr(re(my_lam[0]))
            R_tabs[1].set_parametr(re(my_lam[0]))
            # set theta
            R_tabs[2].set_parametr(re(my_thet[0]))
            R_tabs[3].set_parametr(re(my_thet[0]))

            for i in range(0, len(R_tabs)):
                R_tabs[i].build_curve()


# --------------------------------------------------------------------------
# -------------------- Window forms ----------------------------------------
# --------------------------------------------------------------------------

for i  in [20]:
    print(i)
window['bg'] = '#fafafa'
window.title('SPP reflection viewer')
window.geometry('810x510')
window.resizable(width=False, height=False)

# Frame for ploting
frame_left = Frame(window, relief=RAISED , bd=5)
frame_left.place(x=5, y=5, width=500, height=500)

# Frame fo parametrs
frame_right = Frame(window, relief=RAISED, bd=5)
frame_right.place(x=505, y=5, width=300, height=500)


# -------------------- Left frame - Drawing --------------------------------

# Bottom frame - draw and autodraw
# Create notebook tabs
DrawTabs = ttk.Notebook(frame_left)
DrawTabs.pack(fill=BOTH, expand=True)

R_theta_real = Text(frame_left)
R_theta_imag = Text(frame_left)
R_lambd_real = Text(frame_left)
R_lambd_imag = Text(frame_left)
Grad_profile = Text(frame_left)

DrawTabs.add(R_theta_real, text='    Real R(ϴ)   ') # 0
DrawTabs.add(R_theta_imag, text=' Imaginary R(ϴ) ') # 1
DrawTabs.add(R_lambd_real, text='    Real R(λ)   ') # 2
DrawTabs.add(R_lambd_imag, text=' Imaginary R(λ) ') # 3
DrawTabs.add(Grad_profile, text=' Gradient profile ') # 4

R_tabs = []

# -------------------- Right frame - Parametrs -----------------------------

# Hat - dimension & prism
frm_dim = Frame(frame_right)
frm_dim.pack(fill=X)
lbl_dim= Label(frm_dim, text="          System сharacteristic size = ", font=("Calibri", 10))
lbl_dim.pack(side=LEFT)
dim_var = StringVar()
dim_values = ('1e-5', '1e-6', '1e-7', '1e-8', '1e-9')
spin_dim = Spinbox(frm_dim, values=dim_values, textvariable=dim_var, width=5, command=dim_change)
spin_dim.pack(side=LEFT)

eps_prism_par = Parametr(frame_right, '         Prism ε', 1, 15, 0.1)

# Middle - Layers container
frame_labels = VerticalScrolledFrame(frame_right, relief=SUNKEN, bd=3)
Layers = Layers_container(frame_labels.interior)

# Bottom - Control panel
frame_system_control = Frame(frame_right, bd = 5)
frame_system_control.pack(side=BOTTOM, fill=X)

add_btn = Button(frame_system_control, text="Add layer", font=("Times new roman", 9), command=Layers.add_layer)
add_btn.pack(side=LEFT)

save_btn = Button(frame_system_control, text="Save set", font=("Times new roman", 9), command=save_file)
save_btn.pack(side=RIGHT)

open_btn = Button(frame_system_control, text="Open set", font=("Times new roman", 9), command=open_file)
open_btn.pack(side=RIGHT)

frame_labels.pack(fill=BOTH, expand=True)

# Init first layer and set it to metal
Layers.add_layer()
Layers.switch(invoked='m')

# Create class object in corresponding tab in left frame
experiment_lambda_scale = DoubleVar() #sync lambda in 1 and 2 tab
experiment_lambda_text = StringVar() #sync lambda in 1 and 2 tab
experiment_theta_scale = DoubleVar() #sync theta in 3 and 4 tab
experiment_theta_text = StringVar() #sync theta in 3 and 4 tab

R_tabs.append(Re_theta_tab(R_theta_real, experiment_lambda_scale, experiment_lambda_text))
R_tabs.append(Im_theta_tab(R_theta_imag,  experiment_lambda_scale, experiment_lambda_text))

R_tabs.append(Re_lambda_tab(R_lambd_real,  my_par_theta=experiment_theta_scale, my_par_theta_scl=experiment_theta_text))
R_tabs.append(Im_lambda_tab(R_lambd_imag,  my_par_theta=experiment_theta_scale, my_par_theta_scl=experiment_theta_text))

R_tabs.append(Grad_profile_tab(Grad_profile))
# R_tabs.append(DrawingType_tab(R_lam))


# End program
window.mainloop()