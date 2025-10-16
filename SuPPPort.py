# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:24:08 2022.

Experiment modelling AOF

@author: THzLab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.lib import scimath as SM
from scipy.optimize import minimize_scalar
# from sympy import *
from scipy import interpolate
from types import FunctionType
from abc import ABC, abstractmethod

nm = 1e-9
mkm = 1e-6
_show_SPR_errors = True

def show_SPR_errors():
    global _show_SPR_errors
    _show_SPR_errors = True

def hide_SPR_errors():
    global _show_SPR_errors
    _show_SPR_errors = False

M_cache = {}

def memoize_M(f):
    def decorate(*args):
        if args in M_cache:
            return M_cache[args]
        else:
            M_cache[args] = f(*args)
            return M_cache[args]
    return decorate



# @clock
@memoize_M
def M_matrix(k0, kx0, n_1, n_2, h_1, polarisation = 'p'):
    """Interlayer S matrix - ARRAY."""
    # barchiesi - 2012 - Numerical, k0, kx0 = ext_pars
    k_z1 = SM.sqrt(np.power(k0 * n_1, 2) - kx0)
    k_z2 = SM.sqrt(np.power(k0 * n_2, 2) - kx0)

    if polarisation == 'p':
        r = (k_z1 * n_2**2 - k_z2 * n_1**2) / (k_z1 * n_2**2 + k_z2 * n_1**2)
        t = 2 * (k_z1 * n_1 * n_2) / (k_z1 *n_2**2 + k_z2 *n_1**2)
    if polarisation == 's':
        r = (k_z1 - k_z2) / (k_z1 + k_z2)
        t = 2 * (k_z1) / (k_z1 + k_z2)
    # Stenzel 2016, p. 109
    return np.array([[np.exp(-1j*k_z1*h_1)/t,   np.exp(-1j*k_z1*h_1)*r/t],
                     [np.exp( 1j*k_z1*h_1)*r/t, np.exp( 1j*k_z1*h_1)/t]])



class Layer:
    """Container fpr layer parametrs"""
    S_cache = {}
    gradient_resolution = 100

    def drop_S_cache(self):
        self.S_cache = {}
    
    def memoize_S(f):
        def decorate(*args):
            if (args[1], args[2], args[3], args[4])  in args[0].S_cache:
                return args[0].S_cache[(args[1], args[2], args[3], args[4])]
            else:
                args[0].S_cache[(args[1], args[2], args[3], args[4])] = f(*args)
                return args[0].S_cache[(args[1], args[2], args[3], args[4])]
        return decorate

    def __init__(self, n, thickness, name=None):
        """New layer.

        Parameters
        ----------
        n : float, complex or type
            tepe for layer.
        thickness : float
            thickness of a layer.
        name : strinf, optional
            name of a layer. The default is None.
        """
        self.n = n
        self.thickness = thickness
        self.name = name

    # @memoize_S
    def S_matrix(self, k0, kx0, lining, polarisation):
        """S matrix of layer - lining is TOP, BOTTOM or none"""

        def calc_matrics(nn):
            """One layer matrix with void layers linings"""
            if lining=="MIDLE":
                M1 = M_matrix(k0, kx0, 1, nn, 0, polarisation)
                M2 = M_matrix(k0, kx0, nn, 1, self.thickness, polarisation)
                return M1@M2
            elif lining=="TOP":
                return M_matrix(k0, kx0, nn, 1, self.thickness, polarisation)
            elif lining=="BOTTOM":
                return M_matrix(k0, kx0, 1, nn, 0, polarisation)
            else: print("Wrong type of 'lining' parametr, S matrix is None")

        # Dielectric or metall layers
        if isinstance(self.n, (float, complex, int)):
            return calc_matrics(self.n)
        # Материалы с дисперсией
        elif isinstance(self.n, Dispersion):
            return calc_matrics(self.n.CRI(2 * np.pi / k0))
        # Gradient layer
        elif isinstance(self.n, FunctionType):
            if lining=="MIDLE":
                n_arr = self.n(np.linspace(0,1, self.gradient_resolution))
                M_i = M_matrix(k0, kx0, 1, n_arr[0], 0, polarisation)
                for i in range(0, self.gradient_resolution-1):
                    M_i = M_i@M_matrix(k0, kx0, n_arr[i], n_arr[i+1],
                                       self.thickness/self.gradient_resolution, polarisation)
                return M_i@M_matrix(k0, kx0, n_arr[-1], 1, 0, polarisation)
            elif lining=="TOP":
                return M_matrix(k0, kx0, self.n(1), 1, self.thickness, polarisation)
            elif lining=="BOTTOM":
                return M_matrix(k0, kx0, 1, self.n(0), 0, polarisation)
        # Anisotropic layer
        elif isinstance(self.n, Anisotropic):
            return
        else:
            print("WARNING FATAL ERROR I DONT RECOGNIZE LAYER TYPE!")
            print(type(self.n))


    def __str__(self):
        """Magic representation in string."""
        return 'n=' + str(self.n) + ', d=' + str(round(self.thickness*1e9, 2)) +'nm'        


    def __repr__(self):
        """Magic representation in print(layers)."""
        return '\n - Layer: ' + str(self.n) + ', with d ' + str(self.thickness) + "\n"



class Anisotropic:
    """Anisotropic dielectric layer."""
    main_angle_rad = 0
    n0 = 1
    n1 = 1
    ny_2 = 0
    nz_2 = 0
    nyz = 0
    
    def __init__(self, n0=None, n1=None, anisotropic_angle=None):
        """Anisotropic layer.

        Parameters
        ----------
        n0 : float
            ordinary reflection_data coeficient.
        n1 : float
            extraordinary reflection_data coeficient.
        main_angle_rad : float
            Principle axis angle in degree
        """
        if n0 is not None:
            self.n0 = n0
        if n1 is not None:
            self.n1 = n1
        if anisotropic_angle is not None:
            self.main_angle_rad = np.pi * anisotropic_angle / 180
        else: self.main_angle_rad = 0 # To calculate rafractive indices(setter)

    def kz_dot(self, beta, k0):
        return SM.sqrt(k0**2 * self.ny_2 - beta**2 * self.ny_2 / self.nz_2)

    def K(self):
        return  SM.sqrt(1 - self.nyz**2 / (self.ny_2 * self.nz_2))
    
    def deltaK(self, beta):
        return (beta * self.nyz) / self.nz_2

    def __setattr__(self, name, val):
        """Sync wavelength and k0."""
        self.__dict__[name] = val
        if name == "n0" or name == "n1" or name == "anisotropic_angle":
            if name == "anisotropic_angle":
                self.__dict__['main_angle_rad'] = np.pi * val / 180
            # Equivalent rafractive indices
            self.ny_2 = (self.n0 * np.cos(self.main_angle_rad))**2 \
                + (self.n1 * np.sin(self.main_angle_rad))**2
            self.nz_2 = (self.n0 * np.sin(self.main_angle_rad))**2 \
                + (self.n1 * np.cos(self.main_angle_rad))**2
            self.nyz = (self.n0**2 - self.n1**2) * np.sin(self.main_angle_rad)\
                * np.cos(self.main_angle_rad)

    def kz_plus(self, beta, k0):
        """Kz+."""
        return self.kz_dot(beta, k0) * self.K() + self.deltaK(beta)

    def kz_minus(self, beta, k0):
        """Kz-."""
        return self.kz_dot(beta, k0) * self.K() - self.deltaK(beta)

    def r_in(self, n_prev, beta, k0):
        """r01."""
        a = n_prev**2 / SM.sqrt(n_prev**2 - (beta/k0)**2)
        b = self.p_div_q(beta, k0)
        return - (a - b) / (a + b)

    def r_out(self, n_next, beta, k0):
        """r12."""
        a = self.p_div_q(beta, k0)
        b = n_next**2 / SM.sqrt(n_next**2 - (beta/k0)**2)
        return - (a - b) / (a + b)

    def p_div_q(self, beta, k0):
        """p/q for rij."""
        return SM.sqrt(self.ny_2 * self.nz_2) * self.K() / SM.sqrt(self.nz_2 - (beta/k0)**2)

    def __repr__(self):
        """Magic representation in print(layers)"""
        return "anisotropic, n=(" + str(self.n0) + ", " + str(self.n1) + "), angle=" + str(180*self.main_angle_rad/np.pi)



class Dispersion(ABC):
    """
    Абстрактный базовый класс для материалов с дисперсией.
    
    Основная идея:
    - Каждый материал должен определить внутреннюю функцию self._cri_func,
      которая принимает длину волны (в метрах) и возвращает комплексный n.
    - Публичный метод CRI(wavelength) вызывает эту функцию.
    - Исключение: MaterialDispersion переопределяет CRI, чтобы обрабатывать
      выход за пределы табличного диапазона.
    """
    
    lambda_min_default = 200*nm   # 200 нм
    lambda_max_default = 1200*nm  # 1200 нм

    def __init__(self, name=None):
        self.name = name
        self.lambda_min = self.lambda_min_default
        self.lambda_max = self.lambda_max_default
        # Обязательно задаётся в подклассах
        self._cri_func = None

    def CRI(self, wavelength):
        """
        Публичный интерфейс: возвращает комплексный показатель преломления.
        По умолчанию просто вызывает self._cri_func(wavelength).
        """
        if self._cri_func is None:
            raise NotImplementedError("Подкласс должен задать self._cri_func!")
        return self._cri_func(wavelength)

    def show_CRI(self, lambda_range=None, dpi=None, title=None):
        # ... (остаётся без изменений, как в предыдущем ответе)
        if lambda_range is None:
            lambda_range = np.linspace(self.lambda_min, self.lambda_max, 500)
        else:
            lambda_range = np.asarray(lambda_range)

        n_vals = []
        k_vals = []
        for lam in lambda_range:
            cri = self.CRI(lam)
            n_vals.append(np.real(cri))
            k_vals.append(np.imag(cri))

        if dpi is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(dpi=dpi)
        ax.grid()
        wl_microns = lambda_range * 1e6

        ax.plot(wl_microns, n_vals, label='n')
        ax.plot(wl_microns, k_vals, label='k')

        if title:
            plt.title(title)
        elif self.name:
            plt.title(f'Комплексный показатель преломления {self.name}')
        else:
            plt.title('Комплексный показатель преломления')

        plt.legend(loc='best')
        plt.ylabel('Значение')
        plt.xlabel('Длина волны, мкм')
        plt.show()

    def __repr__(self):
        name_str = f" ({self.name})" if self.name else ""
        return f"{self.__class__.__name__}{name_str}"



class CauchyDispersion(Dispersion):
    """Материал с дисперсией по модели Коши."""
    
    def __init__(self, A=1.0, B=0.0, C=0.0, name=None):
        super().__init__(name)
        self.A, self.B, self.C = A, B, C
        # Задаём внутреннюю функцию — и всё!
        self._cri_func = lambda lam: complex(
            self.A + self.B / (lam * 1e6)**2 + self.C / (lam * 1e6)**4
        )

    def __setattr__(self, name, val):
        super().__setattr__(name, val)
        if all(hasattr(self, attr) for attr in ['A', 'B', 'C']):
            if name in ['A', 'B', 'C']:
                self._cri_func = lambda lam: complex(
                    self.A + self.B / (lam * 1e6)**2 + self.C / (lam * 1e6)**4)



class LorentzDrude(Dispersion):
    """Металл с дисперсией по модели Лоренца-Друде."""
    
    def __init__(self, wp, wt, w0=0, eps_inf=1, amplitude=1, name=None):
        super().__init__(name)
        # Сохраняем "сырые" параметры (могут быть скалярами или массивами)
        self._wp_raw = wp
        self._wt_raw = wt
        self._w0_raw = w0
        self._eps_inf = eps_inf
        self._amplitude_raw = amplitude
        # Инициализируем внутренние массивы и функцию
        self._update_parameters()

    def _update_parameters(self):
        """Приводит параметры к массивам и пересчитывает _cri_func."""
        pi_c = 2 * np.pi * 3e8  # ω = 2πc / λ

        wp = np.atleast_1d(self._wp_raw)
        wt = np.atleast_1d(self._wt_raw)
        w0 = np.atleast_1d(self._w0_raw)
        amplitude = np.atleast_1d(self._amplitude_raw)

        # Расширяем w0 и amplitude до длины wp, если нужно
        if len(w0) == 1 and len(wp) > 1:
            w0 = np.full_like(wp, w0[0], dtype=float)
        if len(amplitude) == 1 and len(wp) > 1:
            amplitude = np.full_like(wp, amplitude[0], dtype=float)

        # Проверка совместимости размеров
        if not (len(wp) == len(wt) == len(w0) == len(amplitude)):
            if _show_SPR_errors:
                print('ОШИБКА! Параметры wp, wt, w0 и amplitude должны иметь одинаковую длину!')
            self._cri_func = lambda lam: 1.0 + 0j
            return

        # Сохраняем нормализованные массивы (опционально, для отладки)
        self._wp = wp
        self._wt = wt
        self._w0 = w0
        self._ampl = amplitude

        # Пересчитываем функцию CRI
        def cri_func(lam):
            omega = pi_c / lam
            eps = self._eps_inf - np.sum([
                self._ampl[i] * self._wp[i]**2 / (omega**2 + 1j * self._wt[i] * omega - self._w0[i]**2)
                for i in range(len(self._wp))
            ])
            return SM.sqrt(eps)

        self._cri_func = cri_func

    # ------------------- Сеттеры -------------------

    @property
    def wp(self):
        return self._wp_raw

    @wp.setter
    def wp(self, value):
        self._wp_raw = value
        self._update_parameters()

    @property
    def wt(self):
        return self._wt_raw

    @wt.setter
    def wt(self, value):
        self._wt_raw = value
        self._update_parameters()

    @property
    def w0(self):
        return self._w0_raw

    @w0.setter
    def w0(self, value):
        self._w0_raw = value
        self._update_parameters()

    @property
    def eps_inf(self):
        return self._eps_inf

    @eps_inf.setter
    def eps_inf(self, value):
        self._eps_inf = value
        self._update_parameters()

    @property
    def amplitude(self):
        return self._amplitude_raw

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude_raw = value
        self._update_parameters()

    # ------------------- Остальной функционал -------------------

    def show_CRI_permittivity(self, lambda_range, dpi=None):
        """График комплексной диэлектрической проницаемости ε = ε' + iε''."""
        eps_re = []
        eps_im = []
        for lam in lambda_range:
            eps = self.CRI(lam)**2
            eps_re.append(np.real(eps))
            eps_im.append(np.imag(eps))

        if dpi is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(dpi=dpi)
        ax.grid()
        wavenumber = 1e-2 / np.array(lambda_range)  # см⁻¹
        ax.plot(wavenumber, eps_re, label="ε'")
        ax.plot(wavenumber, eps_im, label="ε''")
        title = f'Комплексная проницаемость {self.name}' if self.name else 'Комплексная проницаемость'
        plt.title(title)
        plt.legend()
        plt.xlabel('Волновое число, см⁻¹')
        plt.ylabel('ε')
        plt.show()

    def __repr__(self):
        wp_str = [f'{w:.2e}' for w in getattr(self, '_wp', [self._wp_raw])]
        wt_str = [f'{w:.2e}' for w in getattr(self, '_wt', [self._wt_raw])]
        w0_str = [f'{w:.2e}' for w in getattr(self, '_w0', [self._w0_raw])]
        ampl_str = getattr(self, '_ampl', [self._amplitude_raw]).tolist()
        return (f"LorentzDrude({self.name}): wp={wp_str}, wt={wt_str}, "
                f"w0={w0_str}, ampl={ampl_str}")



class MaterialDispersion(Dispersion):
    """
    Материал с дисперсией, заданной табличными данными (n, k) из CSV-файла.
    
    Формат файла базы (например, PermittivitiesBase.csv):
        Индекс,Element,Wavelength,n,k
        0,Au,0.2,0.1,3.5
        1,Au,0.21,0.12,3.48
        ...
    
    Особенности:
    - Длина волны в файле — в микрометрах (мкм).
    - При запросе CRI вне диапазона [min_lam, max_lam] возвращается значение
      на ближайшей границе (экстраполяция константой).
    - Диапазон отображения в show_CRI — автоматически по данным.
    - Поддерживает управление базой материалов.
    """

    def __init__(self, material, base_file=None):
        """
        Инициализация материала из базы.
        
        Parameters
        ----------
        material : str
            Название материала (должно совпадать со значением в столбце 'Element').
        base_file : str, optional
            Путь к CSV-файлу с базой. По умолчанию — "SPPPy/PermittivitiesBase.csv".
        """
        super().__init__(name=material)
        if base_file is None:
            self.base_file = "SPPPy/PermittivitiesBase.csv"
        else:
            self.base_file = base_file

        try:
            Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        except FileNotFoundError:
            if _show_SPR_errors:
                print(f"Файл базы '{self.base_file}' не найден! Используется n = 1 + 0j.")
            self.min_lam_um = 0.2
            self.max_lam_um = 1.2
            self.lambda_min = self.min_lam_um * 1e-6
            self.lambda_max = self.max_lam_um * 1e-6
            self.n_func = lambda x: 1.0
            self.k_func = lambda x: 0.0
            return

        if self.name not in Refraction_data['Element'].values:
            if _show_SPR_errors:
                print(f"Материал '{self.name}' не найден в базе! Используется n = 1 + 0j (воздух).")
            self.min_lam_um = 0.2
            self.max_lam_um = 1.2
            self.lambda_min = self.min_lam_um * 1e-6
            self.lambda_max = self.max_lam_um * 1e-6
            self.n_func = lambda x: 1.0
            self.k_func = lambda x: 0.0
            return

        # Фильтруем данные по материалу
        data = Refraction_data[Refraction_data["Element"] == self.name][["Wavelength", "n", "k"]]
        n_pts = data[['Wavelength', 'n']].dropna().to_numpy()
        k_pts = data[['Wavelength', 'k']].dropna().to_numpy()

        if len(n_pts) == 0 or len(k_pts) == 0:
            if _show_SPR_errors:
                print(f"Нет данных n или k для материала '{self.name}'! Используется n = 1 + 0j.")
            self.min_lam_um = 0.2
            self.max_lam_um = 1.2
            self.lambda_min = self.min_lam_um * 1e-6
            self.lambda_max = self.max_lam_um * 1e-6
            self.n_func = lambda x: 1.0
            self.k_func = lambda x: 0.0
            return

        # Определяем пересечение диапазонов n и k (в мкм)
        self.min_lam_um = max(n_pts[0, 0], k_pts[0, 0])
        self.max_lam_um = min(n_pts[-1, 0], k_pts[-1, 0])

        # Переводим границы в метры для единообразия
        self.lambda_min = self.min_lam_um * 1e-6
        self.lambda_max = self.max_lam_um * 1e-6

        # Создаём интерполяционные функции (линейная интерполяция)
        self.n_func = interpolate.interp1d(n_pts[:, 0], n_pts[:, 1], fill_value="extrapolate")
        self.k_func = interpolate.interp1d(k_pts[:, 0], k_pts[:, 1], fill_value="extrapolate")

    def CRI(self, wavelength):
        """
        Возвращает комплексный показатель преломления n + i*k.
        
        Если длина волны выходит за пределы табличного диапазона,
        используется значение на ближайшей границе.
        
        Parameters
        ----------
        wavelength : float
            Длина волны в метрах.
            
        Returns
        -------
        complex
            n + i*k при заданной длине волны.
        """
        lam_um = wavelength * 1e6  # переводим в микрометры

        # Обрезаем до границ
        if lam_um <= self.min_lam_um:
            lam_um = self.min_lam_um
        elif lam_um >= self.max_lam_um:
            lam_um = self.max_lam_um

        n_val = self.n_func(lam_um)
        k_val = self.k_func(lam_um)
        return complex(n_val, k_val)

    # Метод show_CRI унаследован от Dispersion и использует self.lambda_min/max

    # ------------------- Методы управления базой -------------------

    def add_material(self, element, material_file):
        """
        Добавляет новый материал в базу из CSV-файла (формат refractiveindex.info).
        
        Parameters
        ----------
        element : str
            Название нового материала.
        material_file : str
            Путь к CSV-файлу с данными (n и/или k).
        """
        try:
            Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        except FileNotFoundError:
            Refraction_data = pd.DataFrame(columns=["Element", "Wavelength", "n", "k"])

        if element in Refraction_data['Element'].values:
            if _show_SPR_errors:
                print(f"Материал '{element}' уже существует! Используйте merge_materials.")
            return self

        try:
            df_raw = pd.read_csv(material_file, names=['Wavelength', 'par'])
        except FileNotFoundError:
            if _show_SPR_errors:
                print(f"Файл '{material_file}' не найден!")
            return self

        # Находим разделители 'n' и 'k'
        n_rows = df_raw[df_raw['par'] == 'n'].index
        k_rows = df_raw[df_raw['par'] == 'k'].index

        if len(n_rows) == 0:
            if _show_SPR_errors:
                print("Данные 'n' не найдены в файле!")
            return self

        n_start = n_rows[0] + 1
        k_start = k_rows[0] + 1 if len(k_rows) > 0 else None

        # Извлекаем данные n
        if k_start is not None:
            n_data = df_raw.iloc[n_start:k_start].copy()
        else:
            n_data = df_raw.iloc[n_start:].copy()

        n_data.columns = ['Wavelength', 'n']
        n_data = n_data.astype({'Wavelength': float, 'n': float})

        # Извлекаем данные k (если есть)
        if k_start is not None:
            k_data = df_raw.iloc[k_start:].copy()
            k_data.columns = ['Wavelength', 'k']
            k_data = k_data.astype({'Wavelength': float, 'k': float})
            full_data = pd.merge(n_data, k_data, on='Wavelength', how='outer')
        else:
            n_data['k'] = 0.0
            full_data = n_data

        full_data['Element'] = element
        full_data = full_data[['Element', 'Wavelength', 'n', 'k']].sort_values('Wavelength')

        # Объединяем с базой
        Refraction_data = pd.concat([Refraction_data, full_data], ignore_index=True)
        Refraction_data = Refraction_data.sort_values(["Element", "Wavelength"]).reset_index(drop=True)
        Refraction_data.to_csv(self.base_file)

        if _show_SPR_errors:
            print(f"Материал '{element}' успешно добавлен в базу.")
        return self

    def delete_material(self, material):
        """Удаляет материал из базы по названию."""
        try:
            Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        except FileNotFoundError:
            if _show_SPR_errors:
                print("База не найдена!")
            return self

        if material not in Refraction_data['Element'].values:
            if _show_SPR_errors:
                print(f"Материал '{material}' не найден в базе.")
            return self

        Refraction_data = Refraction_data[Refraction_data['Element'] != material]
        Refraction_data = Refraction_data.sort_values(["Element", "Wavelength"]).reset_index(drop=True)
        Refraction_data.to_csv(self.base_file)

        if _show_SPR_errors:
            print(f"Материал '{material}' удалён из базы.")
        return self

    def merge_materials(self, primary, second, new_name, delete_origin=True):
        """
        Объединяет два материала: берёт данные primary, а из second — только
        те длины волн, которые не пересекаются с primary.
        """
        try:
            Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        except FileNotFoundError:
            if _show_SPR_errors:
                print("База не найдена!")
            return self

        for name in [primary, second]:
            if name not in Refraction_data['Element'].values:
                if _show_SPR_errors:
                    print(f"Материал '{name}' не найден в базе.")
                return self

        if new_name in Refraction_data['Element'].values and not (delete_origin and new_name in [primary, second]):
            if _show_SPR_errors:
                print(f"Материал '{new_name}' уже существует!")
            return self

        # Данные основного материала
        primary_df = Refraction_data[Refraction_data['Element'] == primary][["Wavelength", "n", "k"]]
        prim_min = primary_df["Wavelength"].min()
        prim_max = primary_df["Wavelength"].max()

        # Данные второго материала
        second_df = Refraction_data[Refraction_data['Element'] == second][["Wavelength", "n", "k"]]

        # Берём только то, что вне диапазона primary
        second_outside = second_df[
            (second_df["Wavelength"] < prim_min) | (second_df["Wavelength"] > prim_max)
        ]

        # Объединяем
        merged_df = pd.concat([primary_df, second_outside], ignore_index=True)
        merged_df["Element"] = new_name

        # Удаляем исходные материалы, если нужно
        if delete_origin:
            Refraction_data = Refraction_data[~Refraction_data['Element'].isin([primary, second])]

        Refraction_data = pd.concat([Refraction_data, merged_df], ignore_index=True)
        Refraction_data = Refraction_data.sort_values(["Element", "Wavelength"]).reset_index(drop=True)
        Refraction_data.to_csv(self.base_file)

        if _show_SPR_errors:
            print(f"Материалы '{primary}' и '{second}' объединены в '{new_name}'.")
        return self

    def materials_list(self):
        """Возвращает сводку по всем материалам в базе: диапазон длин волн."""
        try:
            Refraction_data = pd.read_csv(self.base_file, sep=',', index_col=0)
        except FileNotFoundError:
            if _show_SPR_errors:
                print("База не найдена!")
            return pd.DataFrame()

        summary = Refraction_data.groupby('Element')['Wavelength'].agg(['min', 'max'])
        summary.columns = ['λ_min, мкм', 'λ_max, мкм']
        return summary

    def show_base_info(self):
        """Показывает информацию о файле базы."""
        print(f"Файл базы: {self.base_file}")
        try:
            df = pd.read_csv(self.base_file, sep=',', index_col=0)
            print(f"Размер: {df.shape}")
            print(f"Столбцы: {list(df.columns)}")
            print(f"Уникальные материалы: {df['Element'].nunique()}")
        except FileNotFoundError:
            print("Файл не найден.")

    def __repr__(self):
        return f"MaterialDispersion({self.name}): [{self.min_lam_um:.3f}, {self.max_lam_um:.3f}] мкм"



class CompositeDispersion(Dispersion):
    """
    Композитный материал — линейная комбинация нескольких дисперсионных материалов.
    
    CRI(wavelength) = Σ (fraction_i * CRI_i(wavelength))
    
    Доли всегда нормируются так, чтобы их сумма = 1.
    
    Примеры:
    >>> CompositeDispersion([glass, air], [0.7, 0.3]) 
    # 70% стекла + 30% воздуха
    
    >>> CompositeDispersion([glass, air], [7, 3]) 
    # То же самое, что выше — доли нормируются автоматически
    
    >>> CompositeDispersion([glass, glass], [1, 1]) 
    # = glass (усреднение)
    """

    def __init__(self, components, fractions, name=None):
        """
        Инициализация композитного материала.
        
        Parameters
        ----------
        components : Dispersion или list[Dispersion]
            Один материал или список материалов.
        fractions : float или list[float]
            Одна доля или список долей (автоматически нормируются).
        name : str, optional
            Название композита.
        """
        super().__init__(name=name)

        # Превращаем одиночные значения в списки
        if not isinstance(components, (list, tuple)):
            components = [components]
        if not isinstance(fractions, (list, tuple)):
            fractions = [fractions]

        if len(components) == 0 or len(fractions) == 0:
            raise ValueError("Невозможно создать композит из пустых данных")

        # Обрезаем до минимальной общей длины
        min_len = min(len(components), len(fractions))
        if min_len < len(components) or min_len < len(fractions):
            if _show_SPR_errors:
                print(f"Предупреждение: длины components ({len(components)}) и fractions ({len(fractions)}) "
                      f"не совпадают. Используется {min_len} пар.")

        self.components = list(components[:min_len])
        self.fractions = list(fractions[:min_len])

        # Проверяем типы компонентов
        for i, comp in enumerate(self.components):
            if not isinstance(comp, Dispersion):
                raise TypeError(f"Элемент components[{i}] не является экземпляром Dispersion!")

        # ВСЕГДА нормируем доли
        total = sum(self.fractions)
        if total != 0:
            self.fractions = [f / total for f in self.fractions]
        else:
            if _show_SPR_errors:
                print("Предупреждение: сумма долей = 0, невозможно нормировать. Устанавливаю равные доли.")
            n = len(self.fractions)
            self.fractions = [1.0 / n] * n

        # Определяем общий диапазон как пересечение всех диапазонов
        self.lambda_min = max(comp.lambda_min for comp in self.components)
        self.lambda_max = min(comp.lambda_max for comp in self.components)

        # Если пересечение пустое — используем стандартный диапазон
        if self.lambda_min >= self.lambda_max:
            if _show_SPR_errors:
                print("Предупреждение: диапазоны компонентов не пересекаются! Используется стандартный диапазон 200–1200 нм.")
            self.lambda_min = self.lambda_min_default
            self.lambda_max = self.lambda_max_default

    def CRI(self, wavelength):
        """Возвращает линейную комбинацию CRI компонентов."""
        total_cri = 0j
        for frac, comp in zip(self.fractions, self.components):
            total_cri += frac * comp.CRI(wavelength)
        return total_cri

    def __repr__(self):
        comp_names = [comp.name or f"#{i}" for i, comp in enumerate(self.components)]
        details = ", ".join([f"{f:.2f}*{n}" for f, n in zip(self.fractions, comp_names)])
        name_str = f" ({self.name})" if self.name else ""
        return f"CompositeDispersion{name_str}: [{details}]"



# ------------------------------------------------------------------------
# ----------------------- Other functions --------------------------------
# ------------------------------------------------------------------------


def gradient_profile(func, title='Gradient layer profile Shape', dpi=None):
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
    nnn = func(n_range)
    ax.plot(n_range, nnn)
    plt.title(title)
    plt.ylabel('n')
    plt.xlabel('d,%')
    plt.show()


def plot_graph(x, y, title='Reflection data', dpi=None, tir_ang=None,
               label=None, ox_label='ϴ, °', oy_label='R',
               ylim=[0, 1.05], noise=None):
    """
    Parameters:
    x : array(float) or list of arrays(float)
        x coordinates.
    y : array(float) or list of arrays(float)
        y coordinates.
    title : string
        Plot title.
    dpi : int, optional
        DPI for the figure.
    tir_ang : float or list of floats, optional
        Total internal reflection angle(s).
    label : string or list of strings, optional
        Label(s) for the plot(s).
    ox_label : string, optional
        Label for the x-axis.
    oy_label : string, optional
        Label for the y-axis.
    ylim : list of floats, optional
        Limits for the y-axis.
    noise : array(float) or list of arrays(float), optional
        Noise data for fill_between.
    """
    # Преобразуем x и y в списки массивов numpy
    x = [np.array(x)] if not isinstance(x, (list, np.ndarray)) or not isinstance(x[0], (list, np.ndarray)) else [np.array(xi) for xi in x]
    y = [np.array(y)] if not isinstance(y, (list, np.ndarray)) or not isinstance(y[0], (list, np.ndarray)) else [np.array(yi) for yi in y]

    # Проверка на соответствие количества x и y
    if len(x) != len(y):
        print(f"Внимание: количество массивов x ({len(x)}) не совпадает с количеством массивов y ({len(y)}). Будут построены графики для совпадающих пар.")

    # Определяем количество графиков
    num_plots = min(len(x), len(y))

    # Преобразуем label в список, если это не список
    if label is None: labels = [None] * num_plots
    else:  labels = [label] if not isinstance(label, (list, np.ndarray)) else label

    # Проверка на соответствие количества labels и графиков
    if len(labels) < num_plots:
        print(f"Внимание: количество меток ({len(labels)}) меньше количества графиков ({num_plots}). Оставшиеся графики будут без меток.")

    # Создаём фигуру
    if dpi is None: fig, ax = plt.subplots()
    else: fig, ax = plt.subplots(dpi=dpi)
    ax.grid()

    # Строим графики
    for i in range(num_plots):
        current_label = labels[i] if i < len(labels) else None
        ax.plot(x[i], y[i], label=current_label)

    # Легенда
    if any(labels): plt.legend(loc='best')

    # Вертикальные линии
    if tir_ang is not None:
        if isinstance(tir_ang, (list, np.ndarray)):
            for ang in tir_ang:
                plt.axvline(ang, linestyle="--")
        else:
            plt.axvline(tir_ang, linestyle="--")

    # Оформление графика
    plt.title(title)
    plt.ylabel(oy_label)
    plt.xlabel(ox_label)

    # Ограничения по y
    if ylim: plt.ylim(ylim)

    # Noise
    if noise is not None:
        noise = [np.array(noise)] if not isinstance(noise, (list, np.ndarray)) or not isinstance(noise[0], (list, np.ndarray)) else [np.array(noisei) for noisei in noise]
        for i in range(min(num_plots, len(noise))):
            plt.fill_between(x[i], y[i] - noise[i], y[i] + noise[i], color='gray', alpha=0.2)

    plt.show()




def plot_graph_complex(
    R_array,
    title='Reflection data',
    dpi=None,
    labels=None,
    re_limits=None,
    im_limits=None,
):
    """
    Parameters:
    R_array : array(complex) or list of arrays(complex)
        Complex reflection data.
    title : string
        Plot title.
    dpi : int, optional
        DPI for the figure.
    labels : string or list of strings, optional
        Label(s) for the plot(s).
    re_limits : list of 2 floats, optional
        Limits for the real part (Re) as [min, max].
    im_limits : list of 2 floats, optional
        Limits for the imaginary part (Im) as [min, max].
    """
    # Преобразуем R_array в список массивов numpy

    if isinstance(R_array, np.ndarray):
        if np.issubdtype(R_array.dtype, np.complexfloating) and R_array.ndim == 1:
            R_arrays = [R_array]
        else:
            R_arrays = [np.array(R_array)]
    elif isinstance(R_array, list):
        if all(isinstance(r, (np.ndarray, list)) for r in R_array):
            R_arrays = [np.array(r) for r in R_array]
        else:
            R_arrays = [np.array(R_array)]
    else:
        R_arrays = [np.array([R_array])]

    # Преобразуем labels в список, если это не список
    if labels is None:
        labels = [None] * len(R_arrays)
    else:
        labels = [labels] if not isinstance(labels, (list, np.ndarray)) else labels

    # Проверка на соответствие количества R_arrays и labels
    if len(labels) < len(R_arrays):
        print(f"Внимание: количество меток ({len(labels)}) меньше количества графиков ({len(R_arrays)}). Оставшиеся графики будут без меток.")

    # Создаём фигуру
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)

    ax.grid()

    # Строим графики
    scatters = []
    cmap = plt.get_cmap('hsv')  # Используем колормап 'hsv'
    for i, R in enumerate(R_arrays):
        x = np.imag(R)
        y = np.real(R)
        current_label = labels[i] if i < len(labels) else None
        # Цвет точек зависит от их порядкового номера
        c = np.linspace(0, 1, len(R))
        sc = ax.scatter(x, y, c=c, label=current_label, cmap=cmap, vmin=0, vmax=1)
        scatters.append(sc)

    # Легенда
    if any(labels):
        plt.legend(loc='best')

    # Цветовая шкала
    if scatters:
        plt.colorbar(scatters[0], label='Point index')

    # Оформление графика
    plt.title(title)
    plt.ylabel("Re(R)")
    plt.xlabel("Im(R)")

    # Устанавливаем пределы для осей
    if re_limits is not None:
        plt.ylim(re_limits)
    if im_limits is not None:
        plt.xlim(im_limits)

    plt.show()





# def plot_graph_old(x, y, title='Reflection data', dpi=None, tir_ang=None,
#                label=None, ox_label='ϴ, °', oy_label='R',
#                fix_oy_limits=True, ylim=[0, 1.05], noise=None):
#     """Parameters.

#     x : array(float)
#         x cordinates.
#     y : array(float)
#         x cordinates.
#     name : string
#         plot name..
#     """
#     y=np.array(y)

#     if dpi is None:
#         fig, ax = plt.subplots()
#     else:
#         fig, ax = plt.subplots(dpi=dpi)
#     ax.grid()
#     if label is not None:
#         ax.plot(x, y, label=label)
#         plt.legend(loc='best')
#     else:
#         ax.plot(x, y)
#     if tir_ang is not None:
#         plt.axvline(tir_ang, linestyle="--")
#     plt.title(title)
#     if fix_oy_limits:
#         if ylim:
#             plt.ylim(ylim)
#     if noise is not None:
#         noise=np.array(noise)
#         plt.fill_between(x, y - noise, y+noise,
#                       color='gray', alpha=0.2)
        
#     plt.ylabel(oy_label)
#     plt.xlabel(ox_label)
#     plt.show()


# def plot_graph_complex(R_array, title='Reflection data', fix_limits=False, dpi=None,):
#     """Parameters.

#     x : array(float)
#         x cordinates.
#     y : array(float)
#         x cordinates.
#     name : string
#         plot name..
#     """
#     if dpi is None:
#         fig, ax = plt.subplots()
#     else:
#         fig, ax = plt.subplots(dpi=dpi)
#     ax.grid()
    
#     x = []
#     y = []
#     for i in range(0, len(R_array)):
#         x.append(np.real(R_array[i]))
#         y.append(np.imag(R_array[i]))

#     t = np.linspace(0, 2 * np.pi, len(R_array))
#     plt.scatter(t,x,c=y)
#     plt.colorbar()

#     if fix_limits:
#         plt.ylim([-1.05, 1.05])
#         plt.xlim([-1.05, 1.05])

#     plt.title(title)

#     plt.ylabel("Re(R)")
#     plt.xlabel("Im(R)")
#     plt.show()


# def multiplot_graph(plots, title='Reflection data', dpi=None, tir_ang=None,
#                     ox_label='ϴ, °', oy_label='R',
#                     fix_oy_limits=True, ylim=[0, 1.05], noise=None):
#     """Parameters.

#     plots : array(x, y, name, linestyle)
#         like in "Plot_Graph".
#     name : string
#         plot name.
#     tir_ang : int, optional
#         Total internal reflection_data angle. The default is None.
#     """
#     if dpi is None:
#         fig, ax = plt.subplots()
#     else:
#         fig, ax = plt.subplots(dpi=dpi)
#     ax.grid()
#     for i in plots:
#         if len(i) == 2:
#             ax.plot(i[0], i[1])
#         elif len(plots[0]) == 3:
#             ax.plot(i[0], i[1], label=i[2])
#             plt.legend(loc='best')
#         elif len(plots[0]) == 4:
#             ax.plot(i[0], i[1], label=i[2], linestyle=i[3])
#             plt.legend(loc='best')
#         else:
#             if _show_SPR_errors:
#                 print('Not valid array dimension')

    
#     if tir_ang:
#         for i in tir_ang:
#             plt.axvline(i, linestyle="--")

#     if fix_oy_limits:
#         plt.ylim(ylim)
    
#     plt.ylabel(oy_label)
#     plt.xlabel(ox_label)
#     plt.title(title)

#     if noise:
#         x = plots[0][0]
#         y = plots[0][1]
#         ы = np.array([noise]*len(x))
#         plt.fill_between(x, y - ы, y+ы,
#                       color='gray', alpha=0.2)
#     plt.savefig(f'{title}.png')
#     plt.show()


def profile_analyzer(theta_range, reflection_data):
    """Find SPP angle and dispersion halfwidth.

    Parameters.
    reflection_data : array[float]
        reflection_data profile.
    theta_range : range(start, end, seps)
        Range of function definition.

    Returns.
    -------
    xSPPdeg : float
        SPP angle in grad.
    halfwidth : float
        halfwidth.
    """
    div_val = (theta_range.max() - theta_range.min())/len(reflection_data)

    # minimum point - SPP
    yMin = min(reflection_data)
    # print('y_min ',yMin)
    xMin,  = np.where(reflection_data == yMin)[0]
    # print('x_min ',xMin)
    xSPPdeg = theta_range.min() + div_val * xMin

    # first maximum before the SPP
    Left_Part = reflection_data[0:xMin]
    if len(Left_Part) > 0:
        yMax = max(Left_Part)
    else:
        yMax = 1
    left_border = 0
    right_border = reflection_data
    half_height = (yMax-yMin)/2
    point = xMin
    while ((reflection_data[point] < yMin + half_height) &
           (point > 0)):
        point -= 1
    left_border = point
    # print('left hw ', left_border)
    point = xMin
    while ((reflection_data[point] < yMin + half_height) &
           (point < len(reflection_data) - 1)):
        point += 1
    right_border = point
    # print('rigth hw ', right_border)

    halfwidth = div_val * (right_border - left_border)

    # print('xSPPdeg = ', xSPPdeg, 'halfwidth ', halfwidth)
    return xSPPdeg,  halfwidth

def curve_min(x, y):
    """Минимальное положение кривой (x, y)."""
    pos_y = np.min(y)
    pos_x = x[np.where(y == pos_y)[0][0]]
    return (pos_x , pos_y)