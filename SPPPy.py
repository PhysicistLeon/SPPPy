# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:56:28 2021.

Library for calculus in multilayers scheme with gradient layer v 1.0 release
ver 11.08.2021
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

from numpy.lib import scimath as SM
from scipy.optimize import minimize, minimize_scalar
from sympy import *
from scipy import interpolate


class SPP_experiment:
    """
    Experiment class.

    Numeric calculus
    """

    steps = 100  # resolution for gradient layer calculation

    def __init__(self, w, n, d, Gradient=(-1, None)):
        """ Parameters.

        w : float
            light speed normalized angular freqwency.
        n : array[float]
            refractive index array from 0 to N.
        d : array[float]
            layers thickness with d[0]=d[n]=0 on semiinfinite lborder layers.
        Gradient : array[Gradient layer #, gradient in lambda function]]
            gradient layer info. The default is 1.
        """
        self.n = n  # set dielectric permittivities 0-n layer
        self.d = d  # set distances 1-n layer, [0, d1, d2...] must be
        self.w = w  # set wavelength
        self.Gradlayer = Gradient  # set gradient layer parametrs
        
        self.Select_Metals([None]*len(self.n))

        if self.Gradlayer[0] > 0:
            self.n[Gradient[0]] = self.Gradlayer[1](0)
        if len(self.n) != len(self.d):
            print("Warning! Arrays n and d have different length!")
            print("Use array d like [0, d1..., 0] to avoid errors!")

    def display_info(self):
        """Return.

        none
        """
        print("wavelength: ", self.w)
        print("n: ", self.n)
        print("d: ", self.d)
        print(f"Layer #{self.Gradlayer[0]} is gradient")
        n_profile(self.Gradlayer[1])

    def Plot_Grad(self, name=None, dpi=None):
        n_profile(self.Gradlayer[1], name=name, dpi=dpi)

    def Plot_SPP_curve(self, Rmin_curve, plot_2d=False, view_angle=None):

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(xs=Rmin_curve[:, 0], ys=Rmin_curve[:, 1], zs=Rmin_curve[:, 2])
        ax.set_zlim3d(0, max(Rmin_curve[:, 2]))

        ax.set_xlabel('λ , nm')
        ax.set_ylabel('θ, °')
        ax.set_zlabel('R')
        if view_angle is not None:
            ax.view_init(view_angle[0], view_angle[1])

        plt.show()

        if plot_2d:
            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(Rmin_curve[:, 0], Rmin_curve[:, 1])
            ax.set_xlabel('λ , nm')
            ax.set_ylabel('θ, °')
            plt.show()

            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(Rmin_curve[:, 0], Rmin_curve[:, 2])
            ax.set_xlabel('λ , nm')
            ax.set_ylabel('R')
            plt.show()

    def Select_Metals(self, metals_array):
        if len(metals_array) > len(self.n):
            print("Error: metals array is too big! Changes declined")
            return None
        self.Metals_in_unit = [None]*len(self.n)
        for i in range(0, len(metals_array)):
            if metals_array[i] is not None:
                self.Metals_in_unit[i] = Metall_CRI(metals_array[i])
        print(self.Metals_in_unit)

# ----------------------- Calculate profiles -----------------------------

    def R_theta_re(self, degree_range):
        """Parameters.

        degree_range : range(start, end, seps)
            range of function definition.

        Returns
        -------
        Rr : array[float]
            reflection array in range.
        """
        Rr = [np.abs(self.R_deg(np.pi * theta/180, self.w, self.n, self.d, self.Gradlayer))**2
              for theta in degree_range]

        # print(Rr)
        return Rr

    def R_theta_cmpl(self, degree_range):
        """Parameters.

        degree_range : range(start, end, seps)
            range of function definition.

        Returns
        -------
        Rr : array[complex]
            reflection array in range.
        """
        Rr = [self.R_deg(np.pi * theta/180, self.w, self.n, self.d, self.Gradlayer)
              for theta in degree_range]
        return Rr

    def R_lambda_re(self, angle_grad, lambda_range):
        """Parameters.

        angle_grad : float
            angle of calculus in [0 - 90]
        lambda_range : range(start, end, seps)
            range of function definition.

        Returns
        -------
        Rr : array[float]
            reflection array in range.
        """
        Rr = [np.abs(self.R_deg(np.pi * angle_grad/180, 2.0 * np.pi / wave_lambda, self.n, self.d, self.Gradlayer))**2
              for wave_lambda in lambda_range]
        return Rr

    def R_lambda_cmpl(self, angle_grad, lambda_range):
        """Parameters.

        angle_grad : float
            angle of calculus in [0 - 90]
        lambda_range : range(start, end, seps)
            range of function definition.

        Returns
        -------
        Rr : array[complex]
            reflection array in range.
        """
        Rr = [self.R_deg(np.pi * angle_grad/180, 2.0 * np.pi / wave_lambda, self.n, self.d, self.Gradlayer)
              for wave_lambda in lambda_range]
        return Rr

    def GradLayerMatrix(self, theta, w, n, d, Grad):
        """Include output but not input layer.

        Parameters.
        theta : int
            angle to calculate on.
        w : float
            light speed normalized angular freqwency.
        n : array[float]
            refractive index array from 0 to N.
        d : array[float]
            layers thickness with d[0]=d[n]=0 on semiinfinite lborder layers.
        Gradient : array[Gradient layer #, profile shape, shape parametr]]
            gradient layer info.

        Returns.
        -------
        Mtot : matrix [2,2]
            gradient layer matrix to reflection calculus.
        """
        Mtot = np.array([[1, 0], [0, 1]])
        dx = d[Grad[0]]/self.steps
        a = (w * n[0] * np.sin(theta))**2
        ngrad = [self.Gradlayer[1](i/self.steps)
                 for i in range(self.steps)]
        for i in range(1, self.steps):
            ni = ngrad[i-1]
            ni1 = ngrad[i]

            ki = SM.sqrt((w*ni)**2 - a)
            ki1 = SM.sqrt((w*ni1)**2 - a)
            kini1sq = ki * ni1**2
            ki1ni1sq = ki1 * ni**2
            r = (kini1sq - ki1ni1sq) / (kini1sq + ki1ni1sq)
            # if theta*180/np.pi in range (1,40):
            #    print(f'--- > i = {i}, theta = {theta}, dx = {dx}')
            #    print(f'from ni = {ni}, to ni1 = {ni1}')
            kidx = ki*dx
            M = np.array([[np.exp(-1j*kidx) / (1-r),
                           np.exp(-1j*kidx)*r / (1-r)],
                          [np.exp(1j*kidx)*r / (1-r),
                           np.exp(1j*kidx) / (1-r)]])
            Mtot = Mtot@M

        # Output layer
        ni = ni1
        ni1 = n[Grad[0]+1]
        # if theta*180/np.pi in range (1,40):
        #    print(f'Output:')
        #    print(f'from ni = {ni}, to ni1 = {ni1}')

        ki = SM.sqrt((w*ni)**2 - a)
        ki1 = SM.sqrt((w*ni1)**2 - a)
        r = (ki * ni1**2 - ki1 * ni**2) / (ki * ni1**2 + ki1 * ni**2)
        M = np.array([[np.exp(-1j*ki*dx)/(1-r), np.exp(-1j*ki*dx)*r/(1-r)],
                     [np.exp(1j*ki*dx)*r/(1-r), np.exp(1j*ki*dx)/(1-r)]])
        Mtot = Mtot@M
        return Mtot

    def R_deg(self, theta, w, n, d, Gradient):
        """Parameters.

        theta : int
            angle to calculate on.
        w : float
            light speed normalized angular freqwency.
        n : array[float]
            refractive index array from 0 to N.
        d : array[float]
            layers thickness with d[0]=d[n]=0 on semiinfinite lborder layers.
        Gradient : array[Gradient layer #, profile shape, shape parametr]]
            gradient layer info.

        Returns.
        -------
        R : float
            reflection in selected angle.
        """
        
        print(f"n {n}")
        print(f"d {d}")
        print(f"w {w}")
        
        a = np.power(w * n[0]*np.sin(theta), 2)
        k_z = [SM.sqrt(np.power(w*n[i], 2) - a) for i in range(1, len(n))]
        k_z.insert(0, np.sqrt(np.power(w*n[0], 2) - a))

        r = [(k_z[i]*n[i+1]**2-k_z[i+1]*n[i]**2) /
             (k_z[i]*n[i+1]**2+k_z[i+1]*n[i]**2)
             for i in range(0, len(n)-1)]

        # All layers
        M0 = np.array([[1/(1-r[0]), r[0]/(1-r[0])],
                       [r[0]/(1-r[0]), 1/(1-r[0])]])
        for i in range(1, len(n)-1):
            if (i != Gradient[0]):
                Mi = np.array([[np.exp(-1j*k_z[i]*d[i])/(1-r[i]),
                                np.exp(-1j*k_z[i]*d[i])*r[i]/(1-r[i])],
                               [np.exp(1j*k_z[i]*d[i])*r[i]/(1-r[i]),
                                np.exp(1j*k_z[i]*d[i])/(1-r[i])]])
            else:
                Mi = self.GradLayerMatrix(theta, w, n, d, Gradient)
            M0 = M0@Mi

        R = M0[1, 0]/M0[0, 0]
        print(R)
        return R

    def TIR(self):
        """Return.

        TYPE
            Gives angle of total internal reflecion.
        """
        # initial conditions
        warning = None
        TIR_ang = 0
        if (sum(self.d) > 4.0 * np.pi/self.w):
            warning = 'Warning! System too thick to determine\
                total internal reflection angle explicitly!'

        # Otto scheme is when last layer is metal
        if self.n[len(self.n)-1].real < self.n[0].real:
            TIR_ang = np.arcsin(self.n[len(self.n)-1].real/self.n[0].real)
        else:
            # Kretchman scheme is when second layer is metal
            if self.n[1].real > self.n[0].real:
                warning = 'Warning! System too complicated to\
                    determine total internal reflection angle explicitly!'
            for a in range(1, len(self.n)-1):
                if self.n[a].real < self.n[0].real:
                    TIR_ang = np.arcsin(self.n[a].real /
                                        self.n[0].real)
                    break

        # If not found
        if TIR_ang == 0:
            warning = 'Warning! Total internal\
                reflection not occures in that system'
            TIR_ang = np.pi/2

        # if warnings occure
        if warning is not None:
            print(warning)

        return 180*TIR_ang/(np.pi)

    def Difference_homogeniety(self, N_array):
        R_res = len(self.my_reflections_data[0])  # data resolution

        theta_range = self.my_reflections_data[0]
        # take test refractives
        refractive_indexes = list(self.n[0:self.grad_num]) + list(N_array
                                                                  ) + list(self.n[self.grad_num+1:])

        # find discrepancy with new parametrs
        Rr = [(np.abs(self.R_deg(np.pi * theta/180, self.w, refractive_indexes,
                                 self.layers_thickness, (-1, None)))**2) for theta in theta_range]

        Diff = [(Rr[i] - self.my_reflections_data[1][i])
                ** 2 for i in range(R_res)]

        # multiplot_graph([(theta_range, Rr, 'New'),(theta_range, self.my_reflections_data[1], 'Old')], name=sum(Diff))
        return sum(Diff)

    def Restore_Grad_RTheta(self, Data, Resolution):
        """ Grad layer restoring from R(theta)

        Parameters
        ----------
        Data : array
            [R values, theta]
        Resolution : int
            max layers.

        Returns
        -------
        list
            DESCRIPTION.
        """
        # Start with empty
        if Resolution < 2:
            Resolution = 2
        if Resolution > 50:
            Resolution = 50

        self.grad_num = self.Gradlayer[0]
        initial_guess = []
        if self.grad_num < 1 or self.grad_num >= len(self.n)-1:
            print('Gradient layer not detected!')
            return [0]

        self.my_reflections_data = Data
        # from 2 increasing layers count
        for i in range(2, Resolution):
            number_of_layers = i

            # Increase dimension of initial guess
            if len(initial_guess) == 0:
                initial_guess = [1.85]*2
            else:
                x = np.linspace(0, 1, num=len(initial_guess))
                f = sp.interpolate.interp1d(x, initial_guess)
                xnew = np.linspace(0, 1, num=number_of_layers)
                initial_guess = f(xnew)

            # Set new scheme of homogeneous layers
            self.layers_thickness = list(self.d[0:self.grad_num]) + list([self.d[self.grad_num]
                                                                          / number_of_layers]*number_of_layers) + list(self.d[self.grad_num+1:])

            print('d ', self.layers_thickness)
            # Minimize descreapancy with these scheme
            bounds = tuple([(1, 3)]*number_of_layers)
            res = minimize(self.Difference_homogeniety, initial_guess,
                           bounds=bounds, method='Powell')
            initial_guess = res.x
            print('Grad ', initial_guess)

        x = np.linspace(0, 1, num=len(initial_guess))
        f = sp.interpolate.interp1d(x, initial_guess)
        n_profile(f)
        return f


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    def Get_SPP_curve(self, lambda_range, N=None, D=None, Grad=None, MatalList=None):
        if N is None:
            N = self.n
        if D is None:
            D = self.d
        if Grad is None:
            Grad = self.Gradlayer
        if MatalList is None:
            MatalList = self.Metals_in_unit
        
        Rmin_curve = []
        bnds = (47, 51)
        
        for w in lambda_range:
            W0 = 2.0 * np.pi / w


            for i in range(0, len(MatalList)):
                if MatalList[i] is not None:
                    N[i] = MatalList[i].CRI(w)

            # lambda func R(theta) and search of minimum
            def R_dif(th): return np.abs(self.R_deg(
                np.pi * th / 180, W0, N, D, Grad))**2
            theta_min = minimize_scalar(R_dif, bounds=bnds, method='Bounded')

            # minimum value
            Rw_min = np.abs(self.R_deg(
                np.pi * theta_min.x / 180, W0, N, D, Grad))**2

            # print('thetamin ', theta_min.x, ' rmin ', Rw_min)
            bnds = (theta_min.x - 2, theta_min.x + 2)
            Rmin_curve.append([w, theta_min.x, Rw_min])

        return np.array(Rmin_curve)

    def Difference_homogeniety_3d(self, N_array):
        lambda_range = np.array(self.my_reflections_data[:, 0])
        R_res = len(lambda_range)
        # take test refractives

        refractive_indexes = list(self.n[0:self.grad_num]) + list(N_array
                                         ) + list(self.n[self.grad_num+1:])
        Metals_List = list(self.Metals_in_unit[0:self.grad_num]) + [None]*len(N_array
                                         ) + list(self.Metals_in_unit[self.grad_num+1:])
        
        # print(f"len(refractive_indexes) {len(refractive_indexes)}, len(Metals_List) {len(Metals_List)}")

        # find new parametrs
        Rmin_curve = np.array(self.Get_SPP_curve(lambda_range, N=refractive_indexes,
                                                 D=self.layers_thickness, Grad=[-1, lambda x: 1], MatalList=Metals_List))

        # Discrepancy
        Diff = [np.sqrt(((Rmin_curve[i][1] - self.my_reflections_data[i][1])**4 +
                         (Rmin_curve[i][2] - self.my_reflections_data[i][2])**4)) for i in range(R_res)]

        return sum(Diff)

    def Restore_Grad(self, Data, Resolution):
        """ Grad layer restoring from R(theta)

        Parameters
        ----------
        Data : array
            [R values, theta]
        Resolution : int
            max layers.

        Returns
        -------
        list
            DESCRIPTION.
        """
        # Start with empty
        if Resolution < 2:
            Resolution = 2
        if Resolution > 50:
            Resolution = 50

        self.grad_num = self.Gradlayer[0]
        initial_guess = []
        if self.grad_num < 1 or self.grad_num >= len(self.n)-1:
            print('Gradient layer not detected!')
            return [0]

        self.my_reflections_data = Data

        # from 2 increasing layers count
        for i in range(2, Resolution):
            number_of_layers = i

            # Increase dimension of initial guess
            if len(initial_guess) == 0:
                initial_guess = [1.85]*2
            else:
                x = np.linspace(0, 1, num=len(initial_guess))
                f = sp.interpolate.interp1d(x, initial_guess)
                xnew = np.linspace(0, 1, num=number_of_layers)
                initial_guess = f(xnew)

            # Set new scheme of homogeneous layers
            self.layers_thickness = list(self.d[0:self.grad_num]) + list([self.d[self.grad_num]
                     / number_of_layers]*number_of_layers) + list(self.d[self.grad_num+1:])

            # Minimize descreapancy with these scheme
            bounds = tuple([(1, 3)]*number_of_layers)
            res = minimize(self.Difference_homogeniety_3d, initial_guess,
                           bounds=bounds, method='Powell')
            initial_guess = res.x
            print(f'Step = {number_of_layers}, gradient: {initial_guess}')

            # self.Plot_SPP_curve(Rmin_curve, True)

        x = np.linspace(0, 1, num=len(initial_guess))
        f = sp.interpolate.interp1d(x, initial_guess)
        n_profile(f)

        self.refractive_indexes = list(self.n[0:self.grad_num]) + list(initial_guess
                                                                       ) + list(self.n[self.grad_num+1:])
        Rmin_crve = self.Get_SPP_curve(np.array(self.my_reflections_data[:, 0]), N=self.refractive_indexes,
                                       D=self.layers_thickness, Grad=[-1, lambda x: 1])
        self.Plot_SPP_curve(Rmin_crve, plot_2d=True, view_angle=(30, 20))

        return f


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class Metall_CRI:
    def __init__(self, metall, base_file=None):
        if base_file is None:
            base_file = "MetPermittivities.csv"
        self.name = metall

        # Dig for a data
        Refraction_data = pd.read_csv(base_file, sep=',', index_col=0)
        Refraction_data = Refraction_data[Refraction_data["Element"] == metall][[
            "Wavelength", "n", "k"]].to_numpy()

        # Get scope of definition
        self.min_lam = Refraction_data[0][0]
        self.max_lam = Refraction_data[-1][0]

        self.n_func = interpolate.interp1d(
            Refraction_data[:, 0], Refraction_data[:, 1])
        self.k_func = interpolate.interp1d(
            Refraction_data[:, 0], Refraction_data[:, 2])

    def __repr__(self):
        return "<" + self.name + ">"

    def CRI(self, wavelength):
        if wavelength*1e6 >= self.min_lam and wavelength*1e6 <= self.max_lam:
            return (self.n_func(wavelength*1e6) + self.k_func(wavelength*1e6)*1j)
        else:
            print(f"Wavelength is out of bounds!")
            print(f"CRI for {self.name} defined in: [{self.min_lam},{self.max_lam}]µm, and given: {wavelength}µm")
            return None

    def Show_CRI(self):
        fig, ax = plt.subplots()
        ax.grid()
        n_range = np.linspace(self.min_lam, self.max_lam, 100)
        nnn = [self.n_func(j) for j in n_range]
        kkk = [self.k_func(j) for j in n_range]

        ax.plot(n_range, nnn, label='n')
        ax.plot(n_range, kkk, label='k')
        plt.title('Complex refractive index')
        plt.legend(loc='best')
        plt.ylabel('Value')
        plt.xlabel('Wavelength, µm')
        plt.show()

    def Metals_List(self, base_file=None):
        if base_file is None:
            base_file = "MetPermittivities.csv"

        Refraction_data = pd.read_csv(base_file, sep=',', index_col=0)
        
        agg_func_selection = {'Wavelength': ['min', 'max']}
        print(Refraction_data.sort_values(["Element", "Wavelength"], ascending=[True,
                         True]).groupby(['Element']).agg(agg_func_selection))


class SPP_analytic:
    """
    Experiment class.

    Analityc calculus
    """

    def __init__(self, layer_count):
        """ Parameters.

        layer_count : int
            count of layers.
        """
        self.LayerCount = layer_count
        self.r = symbols(f'r(0:{self.LayerCount+1})')
        self.d_kz = symbols(f'd(0:{self.LayerCount+1})')
        self.thetaa = symbols('theta')

    def R_func(self, w=None, n=None, d=None, lambdified=False):
        """
        Get analytic function.
        If parametrs not None - in common form

        Parameters
        ----------
        w : TYPE, optional
            DESCRIPTION. The default is None.
        n : TYPE, optional
            DESCRIPTION. The default is None.
        d : TYPE, optional
            DESCRIPTION. The default is None.
        lambdified : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        from sympy.matrices import Matrix, eye
        m = symbols('M:2:2')
        M = Matrix([[m[0], m[1]], [m[2], m[3]]])
        MSum = eye(2)

        for i in range(0, self.LayerCount+1):
            MSum = expand(simplify(MSum*M.subs([
                (m[0], exp(-I*self.d_kz[i]) / (1-self.r[i])),
                (m[1], exp(-I*self.d_kz[i])*self.r[i]/(1-self.r[i])),
                (m[2], exp(I*self.d_kz[i])*self.r[i]/(1-self.r[i])),
                (m[3], exp(I*self.d_kz[i]) / (1-self.r[i]))])))

        R = simplify(MSum[1, 0]/MSum[0, 0])

        if (w is not None):  # check if numbers are given
            if (n is None) or (d is None):  # check if data is correct
                print('n or d is absent! Substitusion not finished')
            else:
                if len(n) != len(d):  # check data is correct 2
                    print(
                        'n, d or layers count has differetd dimensions! Substitusion not finished')
                else:
                    kx02 = (w * n[0] * sin(self.thetaa))**2

                    R1 = R
                    kz_1 = sqrt((w*n[0])**2 - kx02)
                    for a in range(0, self.LayerCount + 1):
                        kz_0 = kz_1
                        kz_1 = SM.sqrt((w * n[a + 1])**2 - kx02)
                        rr = (kz_0 * n[a + 1]**2 - kz_1 * n[a]**2
                              )/(kz_0 * n[a + 1]**2 + kz_1 * n[a]**2)
                        R1 = R1.subs(
                            [(self.r[a], rr), (self.d_kz[a], d[a]*kz_0)])
                        R = R1
        print(type(R))

        if lambdified:
            return lambdify(self.thetaa, abs(R)**2, sp)
        else:
            return abs(R)**2


# ------------------------------------------------------------------------
# ----------------------- Other functions --------------------------------
# ------------------------------------------------------------------------


# Find SPP angle and dispersion halfwidth
def profile_analyzer(Refl, theta_range):
    """Parameters.
    Refl : array[float]
        reflection profile.
    theta_range : range(start, end, seps)
        Range of function definition.
    Returns.
    -------
    xSPPdeg : float
        SPP angle in grad.
    halfwidth : float
        halfwidth.
    """
    div_val = (theta_range.max() - theta_range.min())/len(Refl)

    # minimum point - SPP
    yMin = min(Refl)
    # print('y_min ',yMin)
    xMin,  = np.where(Refl == yMin)[0]
    # print('x_min ',xMin)
    xSPPdeg = theta_range.min() + div_val * xMin

    # first maximum before the SPP
    Left_Part = Refl[0:xMin]
    if len(Left_Part) > 0:
        yMax = max(Left_Part)
    else:
        yMax = 1
    left_border = 0
    right_border = Refl
    half_height = (yMax-yMin)/2
    point = xMin
    while (Refl[point] < yMin + half_height):
        point -= 1
    left_border = point
    # print('left hw ', left_border)
    point = xMin
    while (Refl[point] < yMin + half_height):
        point += 1
    right_border = point
    # print('rigth hw ', right_border)

    halfwidth = div_val * (right_border - left_border)

    # print('xSPPdeg = ', xSPPdeg, 'halfwidth ', halfwidth)
    return xSPPdeg,  halfwidth


# Draw chosen profiles
def n_profiles(shape, first, last):
    """Parameters.

    shape : int
        number of shape.
    first : int
        first shape paremetr.
    last : int
        first shape paremetr.
    """
    fig, ax = plt.subplots()
    ax.grid()
    n_range = np.linspace(0, 1, 50)

    for N_par in range(first, last+1):
        FFF = n_grad_shapes(shape, N_par)
        nnn = [FFF(j) for j in n_range]
        ax.plot(n_range, nnn, label=f'Var = {N_par}')
    plt.title(f'Gradient layer profiles Shape = {shape}')
    plt.legend(loc='best')
    plt.show()


def n_profile(func, name=None, dpi=None):
    """Parameters.

    func : function
        form of gradient layer in [0,1].

    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    n_range = np.linspace(0, 1, 50)
    nnn = [func(j) for j in n_range]
    ax.plot(n_range, nnn)
    if name is None:
        plt.title('Gradient layer profile Shape')
    else:
        plt.title(name)
    plt.ylabel('n')
    plt.xlabel('d,%')
    plt.show()


def plot_graph(x, y, name='Reflection', tir_ang=None):
    """Parameters.

    x : array(float)
        x cordinates.
    y : array(float)
        x cordinates.
    name : string
        plot name..
    """
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(x, y)
    if tir_ang is not None:
        plt.axvline(tir_ang, linestyle="--")
    plt.title(name)
    plt.ylim([0, 1.05])
    plt.ylabel('R')
    plt.xlabel('ϴ')
    plt.show()


def multiplot_graph(plots, name='Plot', tir_ang=None, dpi=None):
    """Parameters.

    plots : array(x, y, name)
        like in "Plot_Graph".
    name : string
        plot name.
    tir_ang : int, optional
        Total internal reflection angle. The default is None.
    """
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    ax.grid()
    if len(plots[0]) == 3:
        for i in plots:
            ax.plot(i[0], i[1], label=i[2])
    elif len(plots[0]) == 4:
        for i in plots:
            ax.plot(i[0], i[1], label=i[2], linestyle=i[3])
    else:
        plot('Not valid array dimension')

    plt.legend(loc='best')
    if tir_ang is not None:
        plt.axvline(tir_ang, linestyle="--")
    plt.ylabel('R')
    plt.xlabel('ϴ°')
    plt.title(name)
    plt.show()


def n_grad_shapes(shape, N_par):
    """Parameters.

    x : float
        layer coordinate from 0 to 1.
    shape : int
        shape nember.
    N_par : int
        layer parametr.

    Returns.
    -------
    kk : float
        refractive index in x.

    """
    if shape == 1:
        def kk(x): return 1.3 + 0.2*np.sin(3*N_par*(x)-N_par/5)
    elif shape == 2:
        def kk(x): return (1.3 + N_par/10) + (2.33 - 1.79 * (1 + N_par/10))*(x)
    elif shape == 3:
        def kk(x): return 1.1 + (3*N_par/10)*(x-0.5)**2
    elif shape == 4:
        def kk(x): return 1.2 - (5*N_par/10)*(x-0.5)**2
    elif shape == 5:
        def kk(x): return 1.2 + 0.1*np.sin(x*np.pi*2-N_par)
    elif shape == 6:
        def kk(x): return 2/(1 + 4*x/3*(-x+1))
    else:
        def kk(): return 1
    return kk


def main():
    """Return.

    None.
    """
    print('This is library, you can\'t run it :)')


if __name__ == "__main__":
    main()
