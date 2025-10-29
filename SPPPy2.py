# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:56:28 2021.

@author: Tigrus

Library for calculus in multilayers scheme with gradient layer v 2.0 release
ver 11.08.2021
"""


from .SuPPPort import *
import copy
np.seterr(divide='ignore', invalid='ignore')

class ExperimentSPR:
    """Experiment class for numeric calculus."""

    wavelength = 632 * nm  # Unit wavelength
    incidence_angle = 0  # Unit angle
    gradient_resolution = 100  # resolution for gradient layer calculation
    cache = True

    def __init__(self, polarisation='p'):
        """Init empty."""
        self.layers = []
        self.k0 = self.k0_asylum = 2.0 * np.pi / self.wavelength
        self.polarisation = polarisation

    def __setattr__(self, name, val):
        """Sync wavelength and k0."""
        if name == "k0":
            self.__dict__["k0"] = val
            self.__dict__["wavelength"] = 2 * np.pi / val
        elif name == "wavelength":
            self.__dict__["wavelength"] = val
            self.__dict__["k0"] = 2 * np.pi / val
        else: self.__dict__[name] = val

    def __getattr__(self, attrname):
        """Getter for n and d."""
        if attrname == "n":
            val = []
            for L in range(0, len(self.layers)):
                if isinstance(self.layers[L].n, Dispersion):
                    val.append(self.layers[L].n.CRI(self.wavelength))
                elif isinstance(self.layers[L].n, FunctionType):
                    # Gradient layer fubction
                    val.append(self.layers[L].n(0))
                elif isinstance(self.layers[L].n, Anisotropic):
                    # Anisotropic layer fubction
                    val.append(self.layers[L].n.n0)
                else:
                    # Homogenious
                    val.append(self.layers[L].n)
            return val
        if attrname == "d":
            val = [0]
            for L in range(1, len(self.layers)-1):
                val.append(self.layers[L].thickness)
            val.append(0)
            return val

    def save_scheme(self):
        """Conserving scheme parametrs."""
        self.layers_asylum = copy.deepcopy(self.layers)
        self.k0_asylum = self.k0

    def load_scheme(self):
        """Rescuing scheme parametrs."""
        self.layers = copy.deepcopy(self.layers_asylum)
        self.k0 = self.k0_asylum

    # -----------------------------------------------------------------------
    # --------------- Work with layers --------------------------------------
    # -----------------------------------------------------------------------

    def add(self, new_layer):
        """Add one layer.

        Parameters
        ----------
        permittivity : complex, Metall_CRI or lambda
            permittivity for layer.
        thickness : float
            layer thickness.
        """
        self.layers.append(new_layer)

    def delete(self, num):
        """Delete one layer.

        Parameters
        ----------
        num : int
            layer number.
        """
        if num < 0 or num >= len(self.layers):
            print("Deleting layer out of bounds!")
            return
        self.layers.pop(num)

    def insert(self, num, new_layer):
        """Insert layer layer

        Parameters
        ----------
        num : int
            layer number to insert.
        new_layer : [array]
            [permittivity, thickness]
        """
        if num < 0 or num > len(self.layers):
            print("Inserting layer out of bounds! Layer added at the end of the list")
            self.add(new_layer)
        else:
            self.layers.insert(num, new_layer)

    # -----------------------------------------------------------------------
    # --------------- Profiles calculations ---------------------------------
    # -----------------------------------------------------------------------

    # eqs from Stenzel - The Physics of Thin Film Optical Spectra (2016), p 141
    def R(self, ang_range=None, wl_range=None, angle=None, is_complex=False,
          spectral_width=0, spectral_resolution=20):
        """
        Вычисляет коэффициент отражения с учётом конечной спектральной ширины источника.
        
        Поддерживает два экспериментальных режима:
          1. Сканирование по углам (ang_range) при фиксированной номинальной длине волны.
          2. Сканирование по длинам волн (wl_range) при фиксированном угле падения.
          
        При spectral_width > 0 каждая точка результата вычисляется как взвешенное
        усреднение по лоренцианскому спектральному профилю вокруг соответствующей
        центральной длины волны.
        """
        
        # ───────────────────────────────────────────────────────────────
        # 1. Быстрый путь: если спектральная ширина нулевая — просто вызываем
        #    чистую функцию без размытия.
        # ───────────────────────────────────────────────────────────────
        if spectral_width == 0:
            return self._R_basic(
                ang_range=ang_range,
                wl_range=wl_range,
                angle=angle,
                is_complex=is_complex
            )
    
        # ───────────────────────────────────────────────────────────────
        # 2. Проверка корректности входных данных:
        #    - нельзя сканировать и по углам, и по длинам волн одновременно,
        #    - нужно задать хотя бы один из диапазонов.
        # ───────────────────────────────────────────────────────────────
        if ang_range is not None and wl_range is not None:
            raise ValueError("Нельзя одновременно задавать 'ang_range' и 'wl_range'.")
        if ang_range is None and wl_range is None:
            raise ValueError("Необходимо задать либо 'ang_range', либо 'wl_range'.")
    
        # ───────────────────────────────────────────────────────────────
        # 3. Если передан фиксированный угол (например, для спектрального сканирования),
        #    устанавливаем его как текущий угол падения объекта.
        # ───────────────────────────────────────────────────────────────
        if angle is not None:
            self.incidence_angle = angle
    
        # ───────────────────────────────────────────────────────────────
        # 4. Сохраняем исходное состояние объекта (особенно длину волны),
        #    чтобы временные изменения внутри расчётов не повлияли на внешнее поведение.
        #    Это важно, потому что мы будем временно менять self.wavelength.
        # ───────────────────────────────────────────────────────────────
        original_wavelength = self.wavelength
        self.save_scheme()  # сохраняет текущее состояние всей оптической схемы
    
        # ───────────────────────────────────────────────────────────────
        # 5. Используем try/finally, чтобы гарантировать восстановление состояния
        #    даже в случае ошибки (например, исключения в цикле).
        # ───────────────────────────────────────────────────────────────
        try:
    
            # ───────────────────────────────────────────────────────────
            # 6. ВНУТРЕННЯЯ ФУНКЦИЯ: вычисляет одно значение R с учётом спектрального размытия.
            #    Принимает:
            #      - single_angle: угол падения (одно число),
            #      - center_wl: центральная длина волны, вокруг которой делается размытие.
            #    Возвращает:
            #      - комплексное число (если is_complex=True),
            #      - или вещественное |R|² (если is_complex=False).
            # ───────────────────────────────────────────────────────────
            def _R_with_spectral_blur(single_angle, center_wl):
                """
                Вычисляет размытое значение R для одного угла и одной центральной длины волны.
                Использует лоренцианский профиль и вызывает _R_basic для каждой боковой λ.
                """
                # Параметры лоренциана: gamma = FWHM (полная ширина на полувысоте)
                gamma = spectral_width
                # Количество точек для интегрирования: 2 * spectral_resolution - 1
                # (чтобы центральная точка была ровно в середине)
                n_pts = spectral_resolution * 2 - 1
                
                # Диапазон длин волн для усреднения: ±2*gamma охватывает >95% лоренциана
                wl_min = center_wl - 2 * gamma
                wl_max = center_wl + 2 * gamma
                wl_grid = np.linspace(wl_min, wl_max, n_pts)
                
                # Вычисляем веса по формуле лоренциана:
                #   L(λ) = γ / [2π ((λ - λ₀)² + (γ/2)²)]
                weights = gamma / (2 * np.pi * ((wl_grid - center_wl)**2 + (gamma / 2)**2))
                weights /= np.sum(weights)  # нормируем, чтобы сумма весов = 1
    
                # Инициализируем накопитель результата
                if is_complex:
                    total = 0j  # комплексный ноль
                else:
                    total = 0.0  # вещественный ноль
    
                # Перебираем все длины волн из спектрального окна
                for wl, w in zip(wl_grid, weights):
                    # Устанавливаем текущую длину волны для расчёта
                    self.wavelength = wl
                    
                    # Вызываем ЧИСТУЮ функцию _R_basic для одного угла и текущей λ.
                    # Замечание: _R_basic возвращает массив, даже если один элемент.
                    if is_complex:
                        r_val = self._R_basic(
                            ang_range=[single_angle], 
                            wl_range=None, 
                            angle=None, 
                            is_complex=True
                        )
                        total += w * r_val[0]  # r_val[0] — единственное значение
                    else:
                        r_val = self._R_basic(
                            ang_range=[single_angle], 
                            wl_range=None, 
                            angle=None, 
                            is_complex=False
                        )
                        total += w * r_val[0]
    
                return total
    
            # ───────────────────────────────────────────────────────────
            # 7. ОСНОВНОЙ ЦИКЛ: обрабатываем каждую точку измерения.
            #    Выбираем режим в зависимости от того, что задано: углы или длины волн.
            # ───────────────────────────────────────────────────────────
            if ang_range is not None:
                # ─── Режим 1: сканирование по углам ─────────────────────
                # Центральная длина волны — одна и та же для всех углов:
                # это исходное значение self.wavelength (сохранено в original_wavelength).
                results = []
                for theta in ang_range:
                    # Для каждого угла вычисляем размытое R
                    r = _R_with_spectral_blur(single_angle=theta, center_wl=original_wavelength)
                    results.append(r)
                # Возвращаем массив того же типа, что и вход (complex или float)
                return np.array(results, dtype=(complex if is_complex else float))
    
            elif wl_range is not None:
                # ─── Режим 2: сканирование по длинам волн ───────────────
                # Угол фиксирован (либо задан явно через `angle`, либо уже установлен в self).
                # Для каждой длины волны из wl_range — свой центр размытия.
                results = []
                for wl in wl_range:
                    r = _R_with_spectral_blur(single_angle=self.incidence_angle, center_wl=wl)
                    results.append(r)
                return np.array(results, dtype=(complex if is_complex else float))
    
        # ───────────────────────────────────────────────────────────────
        # 8. Восстанавливаем исходное состояние объекта в любом случае
        #    (даже если произошла ошибка). Это критически важно для стабильности.
        # ───────────────────────────────────────────────────────────────
        finally:
            self.load_scheme()  # восстанавливает всё: длину волны, угол, слои и т.д.
            
    def _R_basic(self, ang_range=None, wl_range=None, angle=None, is_complex=False):
        """
        Вычисляет коэффициент отражения без учёта спектральной ширины.
        """
        if angle is not None:
            self.incidence_angle = angle
    
        if ang_range is not None:
            if is_complex:
                return np.array([self.R_deg(theta) for theta in ang_range], dtype=complex)
            else:
                return np.array([np.abs(self.R_deg(theta))**2 for theta in ang_range])
        
        elif wl_range is not None:
            if is_complex:
                return np.array([self.R_deg(wl=wl) for wl in wl_range], dtype=complex)
            else:
                return np.array([np.abs(self.R_deg(wl=wl))**2 for wl in wl_range])
        
        else:
            raise ValueError("Задайте либо 'ang_range', либо 'wl_range'.")

    def T(self, angles=None, wavelengths=None, angle=None, is_complex=False):
          # spectral_width=0, spectral_resolution=20 locked params
        """Representation for every R.

        Parameters
        ----------
        angles : arary, optional
            angles range. The default is None.
        wavelengths : arary, optional
            wavelengths range. The default is None.
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
        if angles is not None:
            if is_complex: return [self.T_deg(theta) for theta in angles]
            else:
                a = []
                for theta in angles:
                    n_0, n_N = self.n[0], self.n[-1]
                    kx0 = self.k0 * np.sin(np.pi*theta/180) * n_0
                    k_z0 = SM.sqrt(np.power(self.k0*n_0, 2) - kx0**2)
                    k_zN = SM.sqrt(np.power(self.k0*n_N, 2) - kx0**2)
                    a.append(np.real(k_zN)/np.real(k_z0) * np.abs(self.T_deg(theta)**2) )
                return a

        # ------- R(lambda) ------------
        elif wavelengths is not None:  
            if is_complex: return [self.T_deg(wl=wl) for wl in wavelengths]
            else:
                sinus = np.sin(np.pi*self.incidence_angle/180)
                a = []
                for wl in wavelengths:
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
        if wl: self.wavelength = wl
        M0 =  self.Transfer_matrix(theta if theta else self.incidence_angle
                                   , self.k0)
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
                if self.polarisation == 'p': a, b = ki * ni1**2, ki1 * ni**2
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
            if self.polarisation == 'p':        
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
            M = self.layers[0].S_matrix(k_0, kx0, 'TOP', self.polarisation)
            for i in range(1, len(self.layers)-1):
                M = M@self.layers[i].S_matrix(k_0, kx0, 'MIDLE', self.polarisation)
            M0 = M@self.layers[len(self.layers)-1].S_matrix(k_0, kx0, 'BOTTOM', self.polarisation)
            return M0
        
        # WITHOUT CACHE! Old calculation without use layer matrix
        else: 
            theta = np.pi * theta / 180
            n, d = self.n, self.d
            kx0_sqrt = k_0 * n[0] * np.sin(theta)
            kx0 = np.power(kx0_sqrt, 2)
            k_z = [SM.sqrt(np.power(k_0*n[i], 2) - kx0) for i in range(0, len(n))]
    
            if self.polarisation == 'p':# All layers for p
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
        for i, value in enumerate(self.layers):
            key = i  # если нужен ключ, используем индекс
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
                    my_label = f"$n_{key}$"
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
            for i, value in enumerate(self.layers):
                key = i
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
                            my_label = f"$k_{key}$"
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
            wl_min = minimize_scalar(lambda x: r_wl_mn(x),
                        bounds=[wl_range[0]/nm, wl_range[-1]/nm], method='Bounded').x
            self.wavelength = wl_min * nm
            Rw_min = np.abs(self.R_deg())**2
            self.load_scheme()
            return wl_min * nm, Rw_min

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
        # bnds = self.curve_angles_range

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
        for i, layer in enumerate(self.layers):
            ax.plot(x1, y1, x2, y2, color='black')
            x1, y1 = [0, 0, 0], [0 - shift, -5 - shift, 0 - shift]
            x2, y2 = [0, 20, 20], [-5 - shift, -5 - shift, 0 - shift]
            shift += 5
    
        text_cords = (5, 2.)
        for layer in self.layers:
            plt.text(text_cords[0], text_cords[1], a)
            text_cords = (text_cords[0], text_cords[1]-5)
        
        plt.text(text_cords[0], text_cords[1], f'λ= {round(self.wavelength * 1e9)}nm')
        plt.show()

    def copy(self):
        newunit = ExperimentSPR()
        newunit.layers = copy.deepcopy(self.layers)
        newunit.wavelength = self.wavelength
        return newunit

def main(): print('This is library, you can\'t run it :)')


if __name__ == "__main__": main()
