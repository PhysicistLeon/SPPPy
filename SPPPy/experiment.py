# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:56:28 2021.

@author: Tigrus

Library for calculus in multilayers scheme with gradient layer v 2.0 release
ver 11.08.2021
"""


from .materials import *
import copy
np.seterr(divide='ignore', invalid='ignore')

class ExperimentSPR:
    """Experiment class for numeric calculus."""

      # Unit wavelength
    gradient_resolution = 100  # resolution for gradient layer calculation
    cache = True

    def __init__(self, polarization='p'):
        """Init empty."""
        self.layers_asylum = self.layers = []
        self.wl_asylum = self.wavelength = 632 * nm
        self.ia_asylum = self.incidence_angle = 0  # Unit angle
        self.polarization = polarization

    def __setattr__(self, name, val):
        """Sync wavelength and k0."""
        if name == "k0":
            self.__dict__["wavelength"] = 2 * np.pi / val

        else: self.__dict__[name] = val

    def __getattr__(self, attrname):
        """Getter for n and d."""
        if attrname == "n":
            val = []
            for layer in self.layers:
                if isinstance(layer.n, DispersionABS):
                    val.append(layer.n.CRI(self.wavelength))
                elif isinstance(layer.n, FunctionType):
                    val.append(layer.n(0))
                elif isinstance(layer.n, Anisotropic):
                    val.append(layer.n.n0)
                else:
                    val.append(layer.n)
            return val
        
        if attrname == "k0":
            return 2.0 * np.pi / self.wavelength

        if attrname == "d":
            val = [0]
            for layer in self.layers[1:-1]:
                val.append(layer.thickness)
            val.append(0)
            return val

    def save_scheme(self):
        """Conserving scheme parametrs."""
        self.layers_asylum = copy.deepcopy(self.layers)
        self.wl_asylum = self.wavelength
        self.ia_asylum = self.incidence_angle

    def load_scheme(self):
        """Rescuing scheme parametrs."""
        self.layers = copy.deepcopy(self.layers_asylum)
        self.wavelength = self.wl_asylum
        self.incidence_angle = self.ia_asylum

    # -----------------------------------------------------------------------
    # --------------- Work with layers --------------------------------------
    # -----------------------------------------------------------------------

    def add(self, new_layer):
        self.layers.append(new_layer)
    
    def delete(self, num):
        if num < 0 or num >= len(self.layers):
            print("Deleting layer out of bounds!")
            return
        del self.layers[num]

    def insert(self, num, new_layer):
        if num < 0 or num > len(self.layers):
            if len(self.layers) > 0:
                print("Inserting layer out of bounds! Layer add in the end of the list")
            self.add(new_layer)
        else:
            self.layers.insert(num, new_layer)

    # -----------------------------------------------------------------------
    # --------------- Profiles calculations ---------------------------------
    # -----------------------------------------------------------------------

    # eqs from Stenzel - The Physics of Thin Film Optical Spectra (2016), p 141
    def R(self, angle_range=None, wl_range=None, angle=None, is_complex=False,
          spectral_width=0, spectral_resolution=20):
        """Representation for every R.

        Parameters
        ----------
        angle_range : arary, optional
            angles range. The default is None.
        wl_range : arary, optional
            wl_range range. The default is None.
        angle : float, optional
            angle for r(lambda). The default is None.
        is_complex : boolean, optional
            return real or complex. The default is false.

        Returns
        -------
        arary
            array of R.
        """
        # Ordinary R
        if spectral_width == 0:
            if angle_range is not None:  # -------------- R(theta) -------------
                if is_complex:
                    return [self.R_deg(theta) for theta in angle_range]
                else:
                    return [np.abs(self.R_deg(theta))**2 for theta in angle_range]
            elif wl_range is not None:  # ------- R(lambda) ------------
                if is_complex:
                    return [self.R_deg(theta = angle if angle else self.incidence_angle,
                                       wl=wl) for wl in wl_range]
                else:
                    return [np.abs(self.R_deg(theta = angle if angle else self.incidence_angle,
                                              wl=wl))**2 for wl in wl_range]
            else: print("Parametrs do not defined!")
        

        # R with spectral width
        if angle_range is not None:
            spectral_function = lambda x: spectral_width / (2 * np.pi * 
                           ((self.wavelength - x)**2 + spectral_width**2 / 4))
            wavelenghts_list = np.linspace(self.wavelength - spectral_width * 2,
                                     self. wavelength + spectral_width * 2,
                                      spectral_resolution * 2 - 1)
            intensities = spectral_function(wavelenghts_list)
            intensities_sum = sum(intensities)
            R_collect = np.zeros(len(angle_range))
            self.save_scheme()
            for i in range(len(wavelenghts_list)):
                self.wavelength = wavelenghts_list[i]
                R_collect = R_collect + [np.abs(self.R_deg(theta))**2 for theta in angle_range] \
                    *(intensities[i]/intensities_sum)
            self.load_scheme()
            return R_collect

    def T(self, angle_range=None, wl_range=None, angle=None, is_complex=False):
          # spectral_width=0, spectral_resolution=20 locked params
        """Representation for every R.

        Parameters
        ----------
        angle_range : arary, optional
            angle_range range. The default is None.
        wl_range : arary, optional
            wl_range range. The default is None.
        angle : float, optional
            angle for r(lambda). The default is None.
        is_complex : boolean, optional
            return real or complex. The default is false.

        Returns
        -------
        arary
            array of T.
        """

        # -------------- T(theta) -------------
        if angle_range is not None:
            if is_complex: return [self.T_deg(theta) for theta in angle_range]
            else:
                a = []
                for theta in angle_range:
                    n_0, n_N = self.n[0], self.n[-1]
                    kx0 = self.k0 * np.sin(np.pi*theta/180) * n_0
                    k_z0 = SM.sqrt(np.power(self.k0*n_0, 2) - kx0**2)
                    k_zN = SM.sqrt(np.power(self.k0*n_N, 2) - kx0**2)
                    a.append(np.real(k_zN)/np.real(k_z0) * np.abs(self.T_deg(theta)**2) )
                return a

        # ------- R(lambda) ------------
        elif wl_range is not None:  
            if is_complex: return [self.T_deg(wl=wl) for wl in wl_range]
            else:
                sinus = np.sin(np.pi*self.incidence_angle/180)
                a = []
                for wl in wl_range:
                    n_0, n_N = self.n[0], self.n[-1]
                    kkk = 2*np.pi/wl
                    kx0 = kkk * sinus * n_0
                    k_z0 = SM.sqrt(np.power(kkk*n_0, 2) - kx0**2)
                    k_zN = SM.sqrt(np.power(kkk*n_N, 2) - kx0**2)
                    a.append(np.real(k_zN)/np.real(k_z0) * np.abs(self.T_deg(wl=wl)**2) )
                return a
            
        else: print("Parametrs do not defined!")



    # eqs from: Byrnes - Multilayer optical calculations (2021), p. 7
    def R_deg(self, theta=None, wl=None):
        M0 =  self.Transfer_matrix(theta if theta else self.incidence_angle
                                   , 2 * np.pi / wl if wl else self.k0)
        if M0[0, 0] == 0:   return 1
        else:   return M0[1, 0]/M0[0, 0]


    def T_deg(self,theta=None, wl=None):
        M0 =  self.Transfer_matrix(theta if theta else self.incidence_angle
                                   ,2 * np.pi / wl if wl else self.k0)
        # print(1/M0[0, 0])
        if M0[0, 0] == 0:   return 0
        else:   return 1 /M0[0, 0]


    def Transfer_matrix(self, theta, k_0):
        """Parameters.

        theta : int
            angle to calculate on.

        Returns.
        -------
        R : float
            complex reflection in selected angle.
        """
        def GradLayerMatrix(theta, n, d, grad_num):
            """BUT CACHE. Include output but not input layer."""
            Mtot = np.array([[1, 0], [0, 1]])
            dx = d[grad_num]/self.gradient_resolution
            kx0 = (k_0 * n[0] * np.sin(theta))**2
            n_range = np.linspace(0, 1, self.gradient_resolution)
            ngrad = self.layers[grad_num].n(n_range)

            for i in range(1, self.gradient_resolution-1):
                ni, ni1= ngrad[i-1], ngrad[i]
                ki = SM.sqrt((k_0*ni)**2 - kx0)
                ki1 = SM.sqrt((k_0*ni1)**2 - kx0)
                if self.polarization == 'p': a, b = ki * ni1**2, ki1 * ni**2
                else: a, b = ki, ki1
                r, t = (a - b) / (a + b), 2*a / (a + b) # SLIGHTLY NOT VALID t
                kidx = ki*dx
                M = np.array([[np.exp(-1j*kidx) / t, np.exp(-1j*kidx)*r / t],
                              [np.exp(1j*kidx)*r / t, np.exp(1j*kidx) / t]])
                Mtot = Mtot@M

            # Output layer
            ni, ni1  = ni1, n[grad_num + 1]
            ki = SM.sqrt((k_0*ni)**2 - kx0)
            ki1 = SM.sqrt((k_0*ni1)**2 - kx0)
            if self.polarization == 'p':        
                r = (ki * ni1**2 - ki1 * ni**2) / (ki * ni1**2 + ki1 * ni**2)
                t = (2 * ki * ni1**2) / (ki * ni1**2 + ki1 * ni**2)
            else:
                r = (ki - ki1) / (ki + ki1)
                t = (2 * ki) / (ki + ki1)
            M = np.array([[np.exp(-1j*ki*dx)/t, np.exp(-1j*ki*dx)*r/t],
                         [np.exp(1j*ki*dx)*r/t, np.exp(1j*ki*dx)/t]])
            Mtot = Mtot@M
            return Mtot

        # CACHE! New calculation with using layer matrix
        if self.cache:
            # print("NEW WAY CALC")
            kx0 = np.power(k_0 * self.n[0] * np.sin(np.pi * theta / 180), 2)
            ext_pars = k_0, kx0
            M = self.layers[0].S_matrix(k_0, kx0, 'TOP', self.polarization)
            for i in range(1, len(self.layers)-1):
                M = M@self.layers[i].S_matrix(k_0, kx0, 'MIDLE', self.polarization)
            M0 = M@self.layers[len(self.layers)-1].S_matrix(k_0, kx0, 'BOTTOM', self.polarization)
            return M0
        
        # WITHOUT CACHE! Old calculation without use layer matrix
        # РУДИМЕНТ ДЛЯ СВЕРКИ c тем, что сверялось со статьями при создании
        else: 
            theta = np.pi * theta / 180
            n, d = self.n, self.d
            kx0_sqrt = k_0 * n[0] * np.sin(theta)
            kx0 = np.power(kx0_sqrt, 2)
            k_z = [SM.sqrt(np.power(k_0*n[i], 2) - kx0) for i in range(0, len(n))]
    
            if self.polarization == 'p':# All layers for p
                r = [(k_z[i]*n[i+1]**2 - k_z[i+1]*n[i]**2) /
                      (k_z[i]*n[i+1]**2 + k_z[i+1]*n[i]**2)
                      for i in range(0, len(n)-1)]
                t = [2*(k_z[i]*n[i]*n[i+1]) /
                      (k_z[i]*n[i+1]**2 + k_z[i+1]*n[i]**2)
                      for i in range(0, len(n)-1)]
                for i in range(1, len(n)-1):
                    if isinstance(self.layers[i].n, Anisotropic):
                        # before layer
                        if not isinstance(self.layers[i - 1].n, Anisotropic):
                            # if previous is anisotropic - r  is modified in prev step
                            r[i-1] = self.layers[i].n.r_in(n[i-1], kx0_sqrt, k_0)
                        # after layer
                        if not isinstance(self.layers[i + 1].n, Anisotropic):
                            # with next isotropic layer
                            r[i] = self.layers[i].n.r_out(n[i+1], kx0_sqrt, k_0)
                        else:
                            # with next anisotropic layer
                            x1 = self.layers[i].n.p_div_q(kx0_sqrt, k_0)
                            x2 = self.layers[i+1].n.p_div_q(kx0_sqrt, k_0)
                            r[i] = (x2 - x1) / (x2 + x1)
                M0 = np.array([[1/t[0], r[0]/t[0]], [r[0]/t[0], 1/t[0]]])
                for i in range(1, len(n)-1):
                    if isinstance(self.layers[i].n, Anisotropic):
                        # 'p' go through and out anisotropic layer 's' dont feels extraordinary n
                        kz_pl = self.layers[i].n.kz_plus(kx0_sqrt, k_0)
                        kz_mn = self.layers[i].n.kz_minus(kx0_sqrt, k_0)
                        Mi = np.array([[np.exp(-1j*kz_pl*d[i])/t[i],
                                        np.exp(-1j*kz_pl*d[i])*r[i]/t[i]],
                                        [np.exp(1j*kz_mn*d[i])*r[i]/t[i],
                                        np.exp(1j*kz_mn*d[i])/t[i]]])
                    elif isinstance(self.layers[i].n, FunctionType):# Gradient layer
                        Mi = GradLayerMatrix(theta, n, d, i)
                    else:# Normal layer
                        Mi = np.array([[np.exp(-1j*k_z[i]*d[i])/t[i],
                                        np.exp(-1j*k_z[i]*d[i])*r[i]/t[i]],
                                        [np.exp(1j*k_z[i]*d[i])*r[i]/t[i],
                                        np.exp(1j*k_z[i]*d[i])/t[i]]])
                    M0 = M0@Mi
                return M0
            # All layers for 's'
            else: 
                r = [(k_z[i] - k_z[i+1]) /
                      (k_z[i] + k_z[i+1])
                      for i in range(0, len(n)-1)]
                t = [2*(k_z[i]) /
                      (k_z[i] + k_z[i+1])
                      for i in range(0, len(n)-1)]
                M0 = np.array([[1/t[0], r[0]/t[0]], [r[0]/t[0], 1/t[0]]])
                for i in range(1, len(n)-1):
                    if isinstance(self.layers[i].n, FunctionType): # Gradient layer
                        Mi = GradLayerMatrix(theta, n, d, i)
                    else:# Normal layer
                        Mi = np.array([[np.exp(-1j*k_z[i]*d[i])/t[i],
                                        np.exp(-1j*k_z[i]*d[i])*r[i]/t[i]],
                                        [np.exp(1j*k_z[i]*d[i])*r[i]/t[i],
                                        np.exp(1j*k_z[i]*d[i])/t[i]]])
                    M0 = M0@Mi
                return M0


    # -----------------------------------------------------------------------
    # --------------- secondary functions -----------------------------------
    # -----------------------------------------------------------------------

    def show_info(self, show_profiles=True):
        """Show set parametrs."""
        word = ' Unit parametrs '
        print(f"{word:-^30}")
        print("k0:", self.k0)
        print("λ: ", self.wavelength)
        print("Ѳ: ", self.incidence_angle)
        print("n: ", self.n)
        print("d: ", self.d)
        if show_profiles:
            self.gradient_profiles(dpi=200)

    def gradient_profiles(self, dpi=None, name=None,
                          save_2_file=None):
        """Parameters.

        func : function
            form of gradient layer in [0,1].
        """
        gradient_found = False
        complex_found = False
        # print(self.layers)
        for idx, value in enumerate(self.layers):
            if isinstance(value.n, FunctionType):
                # if found first
                if not gradient_found:
                    # Initialize plots
                    if dpi is None:
                        fig, ax = plt.subplots()
                    else:
                        fig, ax = plt.subplots(dpi=dpi)
                    ax.grid()
                    n_range = np.linspace(0, 1, self.gradient_resolution)
                    plt.title(name)
                    # remember that plot initialized
                    gradient_found = True
                # draw layer
                if value.name is None:
                    my_label = f"$n_{idx}$"
                else:
                    my_label = value.name
                nnn = value.n(n_range)

                # check for complex vales
                if not complex_found:
                    for i in nnn:
                        if isinstance(i, complex):
                            complex_found = True
                if complex_found:
                    nnn = [np.real(i) for i in nnn]

                ax.plot(n_range, nnn, label=my_label)

        # final
        if gradient_found:
            plt.ylabel('n')
            plt.xlabel('h, отн. ед.')
            plt.legend(loc='best')
            if not complex_found:
                plt.show()
            print("Gradient profiles shown in plots.")
        else:
            print("No gradient layers found.")

        # add complex if found
        if complex_found:
            # init
            n_range = np.linspace(0, 1, 200)
            # fearch for imaginary
            for idx, value in enumerate(self.layers):
                if isinstance(value.n, FunctionType):
                    complex_found = False
                    nnn = value.n(n_range)
                    # check for complex vales
                    for i in nnn:
                        if isinstance(i, complex):
                            complex_found = True
                    if complex_found:
                        nnn = [np.imag(i) for i in nnn]
                        if value.name is None:
                            my_label = f"$k_{idx}$"
                        else:
                            my_label = value.name
                        ax.plot(n_range, nnn, label=my_label)
            plt.legend(loc='best')
            plt.show()
            if save_2_file is not None:
                plt.savefig(save_2_file)

    def pointSPR(self, theta_range=[0,90], wl_range=None): ####***
        """Find minimum = SPR point.

        Parameters
        ----------
        theta_range : array, optional
            angle range to searck. The default is [0,90].

        Returns
        -------
        float
            minimum angle.
        Rw_min : float
            reflettion in minimum angle.
        """

        def r_wl_mn(wll):
            # print(wll)
            self.wavelength = wll * nm
            # print(self.wavelength)
            return np.abs(self.R_deg())**2

        if wl_range is None:
            theta_min = minimize_scalar(lambda x: np.abs(self.R_deg(theta=x))**2,
                        bounds=[theta_range[0], theta_range[-1]], method='Bounded')
            Rw_min = np.abs(self.R_deg(theta_min.x))**2
            return theta_min.x, Rw_min
        else:
            self.save_scheme()

            wl_min = minimize_scalar(lambda x: np.abs(self.R_deg(wl=x*nm))**2,
                        bounds=[wl_range[0]/nm, wl_range[-1]/nm], method='Bounded')
            Rw_min = np.abs(self.R_deg(wl_min.x*nm))**2
            print(f'{wl_min.x*nm} ')
            # Rw_min = np.abs(self.R_deg())**2
            self.load_scheme()
            return wl_min.x*nm, Rw_min

    def TIR(self):
        """Return Gives angle of total internal reflecion."""
        # initial conditions
        warning = None
        TIR_ang = 0
        if (sum(self.d) > 2.0 * self.wavelength):
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

    def get_SPR_curve(self, lambda_range, borders=[30, 70]):
        """Get minimum R(ϴ, λ) for actual set in range.

        Parameters
        ----------
        lambda_range : range
            Range to search.

        Returns
        -------
        array
            [λ, ϴ(SPP), R(SPP)]
        """
        Rmin_curve = []
        # bnds = self.curve_angle_range_range

        self.k0_asylum = self.k0

        for wl in lambda_range:
            self.wavelength = wl
            theta_min, Rw_min = self.pointSPR(borders)
            Rmin_curve.append([wl, theta_min, Rw_min])

        self.k0 = self.k0_asylum
        return np.array(Rmin_curve)

    def plot_SPR_curve(self, Rmin_curve, plot_2d=False, view_angle=None,
                       title=None):
        """Plot R(ϴ, λ).

        Parameters
        ----------
        Rmin_curve : array
            [λ, ϴ(SPP), R(SPP)].
        plot_2d : bool, optional
            If plots R(ϴ) and R(λ)are shown. The default is False.
        view_angle : float, optional
            angle to rotate 3d. The default is None.
        """
        fig = plt.figure(dpi=150)
        ax = fig.gca(projection='3d')
        ax.plot(xs=Rmin_curve[:, 0], ys=Rmin_curve[:, 1], zs=Rmin_curve[:, 2])
        ax.set_zlim3d(0, max(Rmin_curve[:, 2]))

        ax.set_xlabel('λ, nm')
        ax.set_ylabel('θ, °')
        ax.set_zlabel('R')
        if view_angle is not None:
            ax.view_init(view_angle[0], view_angle[1])

        if title: ax.set_title(title)
        plt.show()

        if plot_2d:
            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(Rmin_curve[:, 0], Rmin_curve[:, 1])
            ax.set_xlabel('λ , nm')
            ax.set_ylabel('θ, °')
            if title: ax.set_title(title)
            plt.show()

            fig, ax = plt.subplots()
            ax.grid()
            ax.plot(Rmin_curve[:, 0], Rmin_curve[:, 2])
            ax.set_xlabel('λ , nm')
            ax.set_ylabel('R')
            if title: ax.set_title(title)
            plt.show()

    curve_angles_range = [50, 70]

    def scheme_plot(self, title=None):
        fig, ax = plt.subplots(dpi=200)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis ().set_visible ( False )
        ax.get_yaxis ().set_visible ( False )
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.title(title)
        x1, y1 = [0, 10, 0], [0, 10, 0]
        x2, y2 = [10, 20, 0], [10, 0, 0]
        shift=0
        for a in self.layers:
            ax.plot(x1, y1, x2, y2, color='black')
            x1, y1 = [0, 0, 0], [0 - shift, -5 - shift, 0 - shift]
            x2, y2 = [0, 20, 20], [-5 - shift, -5 - shift, 0 - shift]
            shift += 5
    
        text_cords = (5, 2.)
        for a in self.layers.values():
            plt.text(text_cords[0], text_cords[1], a)
            text_cords = (text_cords[0], text_cords[1]-5)
        
        plt.text(text_cords[0], text_cords[1], f'λ= {round(self.wavelength * 1e9)}nm')
        plt.show()

    def copy(self):
        newunit = ExperimentSPR()
        newunit.layers = copy.deepcopy(self.layers)
        newunit.wavelength = self.wavelength
        newunit.incidence_angle = self.incidence_angle
        return newunit

def main(): print('This is library, you can\'t run it :)')


if __name__ == "__main__": main()
