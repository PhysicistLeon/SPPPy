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
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
from matplotlib.lines import Line2D


nm = 1e-9
mkm = 1e-6
_show_SPR_errors = True
def show_SPR_errors():
    global _show_SPR_errors
    _show_SPR_errors = True
def hide_SPR_errors():
    global _show_SPR_errors
    _show_SPR_errors = False

_show_drop_layers = False
def show_drop_layers():
    global _show_drop_layers
    _show_drop_layers = True
def hide_drop_layers():
    global _show_drop_layers
    _show_drop_layers = False


M_cache = {}

def memoize_M(f):
    def decorate(*args):
        if args in M_cache:
            return M_cache[args]
        else:
            M_cache[args] = f(*args)
            return M_cache[args]
    return decorate

@memoize_M
def M_matrix(k0, kx0, n_1, n_2, h_1, polarization = 'p'):
    """Interlayer S matrix - ARRAY."""
    # barchiesi - 2012 - Numerical, k0, kx0 = ext_pars
    k_z1 = SM.sqrt(np.power(k0 * n_1, 2) - kx0)
    k_z2 = SM.sqrt(np.power(k0 * n_2, 2) - kx0)

    if polarization == 'p':
        r = (k_z1 * n_2**2 - k_z2 * n_1**2) / (k_z1 * n_2**2 + k_z2 * n_1**2)
        t = 2 * (k_z1 * n_1 * n_2) / (k_z1 *n_2**2 + k_z2 *n_1**2)
    if polarization == 's':
        r = (k_z1 - k_z2) / (k_z1 + k_z2)
        t = 2 * (k_z1) / (k_z1 + k_z2)
    # Stenzel 2016, p. 109
    return np.array([[np.exp(-1j*k_z1*h_1)/t,   np.exp(-1j*k_z1*h_1)*r/t],
                     [np.exp( 1j*k_z1*h_1)*r/t, np.exp( 1j*k_z1*h_1)/t]])


class Layer:
    """
    Container for layer parameters in optical structures.
    """
    gradient_resolution: int = 100

    def __init__(self, n: any, thickness: float, name: str = None):
        self.S_cache = {}
        self._last_material_hash = None
        self.name = name
        self.n = n
        self.thickness = thickness

    def _clear_S_cache(self):
        """Clear S_matrix cache and print notification with layer name."""
        self.S_cache.clear()
        global _show_drop_layers
        if _show_drop_layers:
            layer_name = self.name if self.name else "<unnamed>"
            print(f"Кэш S_matrix сброшен для слоя '{layer_name}'")

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self._clear_S_cache()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        self._clear_S_cache()

    def S_matrix(self, k0: complex, kx0: complex, lining: str, polarization: str) -> np.ndarray:
        """
        Calculate the S matrix of the layer with automatic caching and cache invalidation.
        """
        def calc_matrics(nn: complex) -> np.ndarray:
            if lining == "MIDLE":
                M1 = M_matrix(k0, kx0, 1, nn, 0, polarization)
                M2 = M_matrix(k0, kx0, nn, 1, self.thickness, polarization)
                return M1 @ M2
            elif lining == "TOP":
                return M_matrix(k0, kx0, nn, 1, self.thickness, polarization)
            elif lining == "BOTTOM":
                return M_matrix(k0, kx0, 1, nn, 0, polarization)
            else:
                print("Wrong type of 'lining' parameter, S matrix is None")
                return None

        if hasattr(self._n, 'state_hash'):
            current_hash = self._n.state_hash
            if self._last_material_hash != current_hash:
                self.S_cache.clear()
                self._last_material_hash = current_hash

        key = (k0, kx0, lining, polarization, self._thickness)
        if key in self.S_cache:
            return self.S_cache[key]

        # Handle different types of refractive index
        if isinstance(self.n, (float, complex, int)):
            result = calc_matrics(self.n)
        elif isinstance(self.n, DispersionABS):
            result = calc_matrics(self.n.CRI(2 * np.pi / k0))
        elif isinstance(self.n, FunctionType):
            if lining == "MIDLE":
                n_arr = self.n(np.linspace(0, 1, self.gradient_resolution))
                M_i = M_matrix(k0, kx0, 1, n_arr[0], 0, polarization)
                for i in range(0, self.gradient_resolution - 1):
                    M_i = M_i @ M_matrix(k0, kx0, n_arr[i], n_arr[i + 1],
                                         self.thickness / self.gradient_resolution, polarization)
                result = M_i @ M_matrix(k0, kx0, n_arr[-1], 1, 0, polarization)
            elif lining == "TOP":
                result = M_matrix(k0, kx0, self.n(1), 1, self.thickness, polarization)
            elif lining == "BOTTOM":
                result = M_matrix(k0, kx0, 1, self.n(0), 0, polarization)
        elif isinstance(self.n, Anisotropic):
            # АНИЗОТРОПНЫЙ СЛОЙ С КЕШЕМ: обёрнут нулевыми воздушными слоями
            # Соседи всегда воздух (n=1.0), как и во всех S-матрицах
            
            if lining == "MIDLE":
                # Структура: [воздух→анизо, d=0] @ [анизо→воздух, d]
                
                # Входной интерфейс: воздух (n=1) → анизотроп
                kx0_sqrt = SM.sqrt(kx0)
                r_in_val = self.n.r_in(1.0, kx0_sqrt, k0)
                M1 = np.array([[1.0, r_in_val],
                              [r_in_val, 1.0]])
                
                # Распространение в анизотропе + выход в воздух
                r_out_val = self.n.r_out(1.0, kx0_sqrt, k0)
                kz_pl = self.n.kz_plus(kx0_sqrt, k0)
                kz_mn = self.n.kz_minus(kx0_sqrt, k0)
                
                phase_pl = np.exp(-1j * kz_pl * self.thickness)
                phase_mn = np.exp(1j * kz_mn * self.thickness)
                
                M2 = np.array([[phase_pl, phase_pl * r_out_val],
                              [phase_mn * r_out_val, phase_mn]])
                
                result = M1 @ M2
                
            elif lining == "TOP":
                # TOP: выход из анизотропа в воздух (без входного интерфейса)
                kx0_sqrt = SM.sqrt(kx0)
                r_out_val = self.n.r_out(1.0, kx0_sqrt, k0)
                
                kz_pl = self.n.kz_plus(kx0_sqrt, k0)
                kz_mn = self.n.kz_minus(kx0_sqrt, k0)
                
                phase_pl = np.exp(-1j * kz_pl * self.thickness)
                phase_mn = np.exp(1j * kz_mn * self.thickness)
                
                result = np.array([[phase_pl, phase_pl * r_out_val],
                                  [phase_mn * r_out_val, phase_mn]])
                
            elif lining == "BOTTOM":
                # BOTTOM: вход из воздуха в анизотроп (без выходного интерфейса)
                kx0_sqrt = SM.sqrt(kx0)
                r_in_val = self.n.r_in(1.0, kx0_sqrt, k0)
                
                result = np.array([[1.0, r_in_val],
                                  [r_in_val, 1.0]])
            
            else:
                print("Wrong type of 'lining' parameter for Anisotropic, S matrix is None")
                result = None

        else:
            print("WARNING FATAL ERROR I DONT RECOGNIZE LAYER TYPE!")
            print(type(self.n))
            result = None

        if result is not None:
            self.S_cache[key] = result
        return result

    def __str__(self) -> str:
        return 'n=' + str(self.n) + ', d=' + str(round(self.thickness * 1e9, 2)) + 'nm'

    def __repr__(self) -> str:
        return '\n - Layer: ' + str(self.n) + ', with d ' + str(self.thickness) + "\n"


class Anisotropic:
    """Anisotropic dielectric layer."""
    
    def __init__(self, n0=None, n1=None, anisotropic_angle=None):
        """Anisotropic layer.

        Parameters
        ----------
        n0 : float
            ordinary reflection_data coeficient.
        n1 : float
            extraordinary reflection_data coeficient.
        anisotropic_angle : float
            Principle axis angle in degrees
        """
        self._state_hash = None

        if n0 is not None:
            self.n0 = n0
        if n1 is not None:
            self.n1 = n1
        if anisotropic_angle is not None:
            self.anisotropic_angle = anisotropic_angle  # градусы, через __setattr__
        else:
            self.anisotropic_angle = 0  # градусы, через __setattr__

    def __setattr__(self, name: str, val: any) -> None:
        # Сбрасываем хэш только при изменении физически значимых параметров
        if name in ("n0", "n1", "anisotropic_angle"):
            self.__dict__['_state_hash'] = None
    
        # Записываем значение
        self.__dict__[name] = val
    
        # Пересчёт внутренних параметров при изменении n0, n1 или угла
        if name in ("n0", "n1", "anisotropic_angle"):
            if hasattr(self, 'n0') and hasattr(self, 'n1'):
                main_angle_rad = np.pi * val / 180
                self.ny_2 = (self.n0 * np.cos(main_angle_rad))**2 + (self.n1 * np.sin(main_angle_rad))**2
                self.nz_2 = (self.n0 * np.sin(main_angle_rad))**2 + (self.n1 * np.cos(main_angle_rad))**2
                self.nyz = (self.n0**2 - self.n1**2) * np.sin(main_angle_rad) * np.cos(main_angle_rad)

    

    @property
    def state_hash(self) -> int:
        if self._state_hash is None:
            self._state_hash = hash((self.n0, self.n1, self.anisotropic_angle))
        return self._state_hash

    def kz_dot(self, beta, k0):
        return SM.sqrt(k0**2 * self.ny_2 - beta**2 * self.ny_2 / self.nz_2)

    def K(self):
        return  SM.sqrt(1 - self.nyz**2 / (self.ny_2 * self.nz_2))
    
    def deltaK(self, beta):
        return (beta * self.nyz) / self.nz_2

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
        return f"Anisotropic(n0={self.n0}, n1={self.n1}, angle_deg={self.anisotropic_angle:.2f})"



class DispersionABS(ABC):
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
        self._state_hash = None

    @property
    def state_hash(self) -> int:
        """Хэш состояния дисперсии.
        Подклассы должны переопределить для отслеживания своих параметров.
        """
        raise NotImplementedError("Подклассы должны реализовать state_hash")

    def CRI(self, wavelength):
        """
        Публичный интерфейс: возвращает комплексный показатель преломления.
        По умолчанию просто вызывает self._cri_func(wavelength).
        """
        if self._cri_func is None:
            raise NotImplementedError("Подкласс должен задать self._cri_func!")
        return self._cri_func(wavelength)

    def show_CRI(self, lambda_range=None, dpi=None, title=None):
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
    
        # Преобразуем в массивы для удобства
        n_vals = np.array(n_vals)
        k_vals = np.array(k_vals)
    
        if dpi is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(dpi=dpi)
        ax.grid()
        wl_microns = lambda_range * 1e6
    
        # Всегда рисуем n
        ax.plot(wl_microns, n_vals, label='n')
    
        # Рисуем k, только если он не тождественно нулевой
        if not np.allclose(k_vals, 0.0, atol=1e-10):
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



class CauchyDispersion(DispersionABS):
    """Материал с дисперсией по модели Коши."""

    def __init__(self, A=1.0, B=0.0, C=0.0, name=None):
        super().__init__(name)
        self.A, self.B, self.C = A, B, C
        # Задаём внутреннюю функцию — и всё!
        self._cri_func = lambda lam: complex(
            self.A + self.B / (lam * 1e6)**2 + self.C / (lam * 1e6)**4)

    @property
    def state_hash(self) -> int:
        if self._state_hash is None:
            self._state_hash = hash((
                self.A,
                self.B,
                self.C
            ))
        return self._state_hash

    def __setattr__(self, name, val):
        # Сбрасываем хэш только при изменении релевантных атрибутов
        if name in ('A', 'B', 'C'):
            self.__dict__['_state_hash'] = None
        super().__setattr__(name, val)
        # Пересчитываем _cri_func только при изменении A, B, C
        if name in ('A', 'B', 'C'):
            self._cri_func = lambda lam: complex(
                self.A + self.B / (lam * 1e6)**2 + self.C / (lam * 1e6)**4)



class LorentzDrudeDispersion(DispersionABS):
    """Металл с дисперсией по модели Лоренца-Друде."""
    
    def __init__(self, wp, wt, w0=0, amplitude=1, eps_inf=1, name=None):
        super().__init__(name)
        # Конвертируем входные данные в массивы
        self._wp = TrackedArray(np.atleast_1d(wp))
        self._wt = TrackedArray(np.atleast_1d(wt))
        self._w0 = TrackedArray(np.atleast_1d(w0))
        self._ampl = TrackedArray(np.atleast_1d(amplitude))
        self._eps_inf = eps_inf
        self._state_hash = None
        self._update_parameters()

    @property
    def state_hash(self):
        """
        Проверяет хэши массивов и свой хэш.
        Если хотя бы один хэш сброшен (равен None), пересчитывает все хэши и обновляет параметры.
        Возвращает кортеж хэшей всех параметров.
        """
        # Проверяем, не сброшен ли хэш хотя бы одного массива или свой хэш
        if (self._wp._hash is None or
            self._wt._hash is None or
            self._w0._hash is None or
            self._ampl._hash is None or
            self._state_hash is None):
    
            # Пересчитываем хэш всех массивов и свой хэш
            self._state_hash = (
                self._wp.hash,    # Пересчитывает хэш, если он None
                self._wt.hash,
                self._w0.hash,
                self._ampl.hash,
                hash(self._eps_inf)
            )
            self._update_parameters()  # Пересчитываем параметры
    
        return self._state_hash

    @property
    def eps_inf(self):
        return self._eps_inf
    
    @eps_inf.setter
    def eps_inf(self, val):
        self._eps_inf = val
        self._state_hash = None  # заставит state_hash пересобрать cri_func

    @property
    def wp(self):
        """Плазменные частоты осцилляторов (массив или скаляр, внешнее API)."""
        return self._wp
    
    @wp.setter
    def wp(self, val):
        self._wp = TrackedArray(np.atleast_1d(val))
        self._state_hash = None  # заставит state_hash пересчитать хэши и CRI
    
    
    @property
    def wt(self):
        """Ширины (затухания) осцилляторов."""
        return self._wt
    
    @wt.setter
    def wt(self, val):
        self._wt = TrackedArray(np.atleast_1d(val))
        self._state_hash = None
    
    @property
    def w0(self):
        """Резонансные частоты осцилляторов."""
        return self._w0
    
    @w0.setter
    def w0(self, val):
        self._w0 = TrackedArray(np.atleast_1d(val))
        self._state_hash = None
    
    
    @property
    def ampl(self):
        """Амплитуды осцилляторов."""
        return self._ampl
    
    @ampl.setter
    def ampl(self, val):
        self._ampl = TrackedArray(np.atleast_1d(val))
        self._state_hash = None

    def _update_parameters(self):
        """
        Пересчитывает функцию CRI, используя минимальную длину массивов.
        Обеспечивает корректную работу даже при разной длине входных массивов.
        """
        pi_c = 2 * np.pi * 3e8  # ω = 2πc / λ
    
        # Определяем минимальную длину среди всех массивов
        min_len = min(len(self._wp), len(self._wt), len(self._w0), len(self._ampl))
    
        # Функция для вычисления комплексного показателя преломления
        def cri_func(lam):
            omega = pi_c / lam
            # Суммируем вклад каждого осциллятора (только по минимальной длине)
            eps = self._eps_inf - np.sum([
                self._ampl[i] * self._wp[i]**2 /
                (omega**2 + 1j * self._wt[i] * omega - self._w0[i]**2)
                for i in range(min_len)
            ])
            return SM.sqrt(eps)  # Возвращаем комплексный показатель преломления
    
        # Сохраняем функцию для дальнейшего использования
        self._cri_func = cri_func

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
        wp_str = [f'{w:.2e}' for w in self._wp.__array__()]
        wt_str = [f'{w:.2e}' for w in self._wt.__array__()]
        w0_str = [f'{w:.2e}' for w in self._w0.__array__()]
        ampl_str = self._ampl.__array__().tolist()
        return (f"LorentzDrude({self.name}): wp={wp_str}, wt={wt_str}, "
                f"w0={w0_str}, ampl={ampl_str}")



class MaterialDispersion(DispersionABS):
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
            
        Примечание
        ----------
        API работает в метрах (SI система, нанометры).
        - lambda_min, lambda_max — видимые параметры в метрах
        - Микрометры (для CSV и интерполяции) вычисляются локально в методах
        """
        super().__init__(name=material)
        
        if base_file is None:
            self.base_file = "PermittivitiesBase.csv"
        else:
            self.base_file = base_file
    
        try:
            Refraction_data = self._load_base()
        except FileNotFoundError:
            if _show_SPR_errors:
                print(f"Файл базы '{self.base_file}' не найден! Используется n = 1 + 0j.")
            self.lambda_min = 200e-9
            self.lambda_max = 1200e-9
            self.n_func = lambda x: 1.0
            self.k_func = lambda x: 0.0
            return
    
        if self.name not in Refraction_data['Element'].values:
            if _show_SPR_errors:
                if self.name != 'Air':
                    print(f"Материал '{self.name}' не найден в базе! Используется n = 1 + 0j (воздух).")
            self.lambda_min = 200e-9
            self.lambda_max = 1200e-9
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
            self.lambda_min = 200e-9
            self.lambda_max = 1200e-9
            self.n_func = lambda x: 1.0
            self.k_func = lambda x: 0.0
            return
    
        # Определяем пересечение диапазонов n и k
        # CSV в микрометрах, сразу конвертируем в метры (SI)
        self.lambda_min = max(n_pts[0, 0], k_pts[0, 0]) * 1e-6
        self.lambda_max = min(n_pts[-1, 0], k_pts[-1, 0]) * 1e-6
    
        # Создаём интерполяционные функции (линейная интерполяция)
        self.n_func = interpolate.interp1d(n_pts[:, 0], n_pts[:, 1], fill_value="extrapolate")
        self.k_func = interpolate.interp1d(k_pts[:, 0], k_pts[:, 1], fill_value="extrapolate")


    def _load_base(self):
        """
        Загружает базу материалов из self.base_file.
        - Если файл не найден — пробрасывает FileNotFoundError.
        - Если файл найден, но нет столбца 'Source' — добавляет его со значением ''.
        - Никакой сортировки, фильтрации или очистки не выполняется.
        Возвращает pd.DataFrame с колонками, включая 'Source'.
        """
        df = pd.read_csv(self.base_file, sep=',', index_col=0)

        if "Source" not in df.columns:
            df["Source"] = ""
        else:
            df["Source"] = df["Source"].fillna("").astype(str)

        return df

    def _save_base(self, df):
        """
        Сохраняет базу с нормализацией:
        - сортировка по Element, затем Wavelength
        - сброс индекса
        - сохранение в CSV
        """
        # Убедимся, что обязательные столбцы есть (защита от багов)
        required = {"Element", "Wavelength", "n", "k", "Source"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Отсутствуют столбцы при сохранении: {missing}")
    
        # Сортировка и сброс индекса — как в оригинальных методах
        df_sorted = df.sort_values(["Element", "Wavelength"]).reset_index(drop=True)
    
        # Сохраняем
        df_sorted.to_csv(self.base_file, sep=',')
        
    @property
    def state_hash(self) -> int:
        """Хэш зависит от диапазона длин волн в метрах"""
        if self._state_hash is None:
            self._state_hash = hash((self.lambda_min, self.lambda_max))
        return self._state_hash

    def __setattr__(self, name, val):
        """Перехватывает изменения видимых параметров"""
        # Сбрасываем хэш при изменении диапазона длин волн
        if name in ('lambda_min', 'lambda_max'):
            self.__dict__['_state_hash'] = None
        
        super().__setattr__(name, val)

        
    def CRI(self, wavelength):
        """
        Возвращает комплексный показатель преломления n + i*k.

        wavelength в метрах. Внутри используются микрометры для интерполяции.
        """
        lam_um = wavelength * 1e6

        # Вычисляем границы в микрометрах из видимых параметров
        min_lam_um = self.lambda_min * 1e6
        max_lam_um = self.lambda_max * 1e6

        # Обрезаем до границ
        if lam_um <= min_lam_um:
            lam_um = min_lam_um
        elif lam_um >= max_lam_um:
            lam_um = max_lam_um

        n_val = self.n_func(lam_um)
        k_val = self.k_func(lam_um)

        return complex(n_val, k_val)

    # Метод show_CRI унаследован от DispersionABS и использует self.lambda_min/max

    # ------------------- Методы управления базой -------------------

    def add_material(self, element, material_file, source=''):
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
            Refraction_data = self._load_base()
        except FileNotFoundError:
            Refraction_data = pd.DataFrame(columns=["Element", "Wavelength", "n", "k", "Source"])

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
            n_data = df_raw.iloc[n_start:k_start-1].copy()
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
        full_data['Source'] = source
        full_data = full_data.sort_values('Wavelength')

        # Объединяем с базой
        Refraction_data = pd.concat([Refraction_data, full_data], ignore_index=True)
        self._save_base(Refraction_data)  # ← вся сортировка и запись внутри

        if _show_SPR_errors:
            print(f"Материал '{element}' успешно добавлен в базу.")
        return self

    def delete_material(self, material):
        """Удаляет материал из базы по названию."""
        try:
            Refraction_data = self._load_base()
        except FileNotFoundError:
            if _show_SPR_errors:
                print("База не найдена!")
            return self

        if material not in Refraction_data['Element'].values:
            if _show_SPR_errors:
                print(f"Материал '{material}' не найден в базе.")
            return self

        Refraction_data = Refraction_data[Refraction_data['Element'] != material]
        self._save_base(Refraction_data)

        if _show_SPR_errors:
            print(f"Материал '{material}' удалён из базы.")
        return self

    def merge_materials(self, primary, second, new_name, source='', delete_origin=False):
        """
        Объединяет два материала: берёт данные primary, а из second — только
        те длины волн, которые не пересекаются с primary.
        """
        try:
            Refraction_data = self._load_base()
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
        merged_df["Source"] = source

        # Удаляем исходные материалы, если нужно
        if delete_origin:
            Refraction_data = Refraction_data[~Refraction_data['Element'].isin([primary, second])]

        Refraction_data = pd.concat([Refraction_data, merged_df], ignore_index=True)
        self._save_base(Refraction_data)

        if _show_SPR_errors:
            print(f"Материалы '{primary}' и '{second}' объединены в '{new_name}'.")
        return self

    def materials_list(self):
        try:
            Refraction_data = self._load_base()
        except FileNotFoundError:
            if _show_SPR_errors:
                print("База не найдена!")
            return pd.DataFrame()
    
        # Группируем и агрегируем: min/max длины волны + источник
        summary = Refraction_data.groupby('Element').agg(
            λ_min=('Wavelength', 'min'),
            λ_max=('Wavelength', 'max'),
            Source=('Source', 'first')  # или 'last', или проверка на уникальность
        )
        return summary

    def __repr__(self):
        return f"MaterialDispersion({self.name}): [{self.lambda_min*1e9:.1f}, {self.lambda_max*1e9:.1f}] нм"



class CompositeDispersion(DispersionABS):
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
        Parameters
        ----------
        components : sequence of DispersionABS
            Список/кортеж материалов (объектов, имеющих метод CRI и state_hash).
        fractions : sequence of float
            Доли компонентов (любые положительные числа, будут нормированы).
        name : str, optional
            Имя композита (для вывода).
        """
        super().__init__(name=name)

        # Приводим компоненты к списку
        self.components = list(components)

        if len(self.components) == 0:
            raise ValueError("CompositeDispersion: components list cannot be empty")

        # Проверяем, что все компоненты — DispersionABS
        for c in self.components:
            if not isinstance(c, DispersionABS):
                raise TypeError("All components must be instances of DispersionABS or its subclasses.")

        # Приводим доли к массиву и нормируем
        fr = np.asarray(fractions, dtype=float)
        if fr.ndim != 1 or len(fr) != len(self.components):
            raise ValueError("fractions must be 1D and have the same length as components")
        if fr.sum() == 0:
            raise ValueError("Sum of fractions cannot be zero")

        fr = fr / fr.sum()
        # Отслеживаемый массив долей
        self._fractions = TrackedArray(fr)

        # Диапазон композита — пересечение диапазонов всех компонентов
        self.lambda_min = max(comp.lambda_min for comp in self.components)
        self.lambda_max = min(comp.lambda_max for comp in self.components)

        self._state_hash = None

    @property
    def fractions(self):
        """Массив долей компонентов (нормирован)."""
        return self._fractions

    @fractions.setter
    def fractions(self, val):
        """Замена долей компонентов."""
        fr = np.asarray(val, dtype=float)
        if fr.ndim != 1 or len(fr) != len(self.components):
            raise ValueError("fractions must be 1D and have the same length as components")
        if fr.sum() == 0:
            raise ValueError("Sum of fractions cannot be zero")
        fr = fr / fr.sum()
        self._fractions = TrackedArray(fr)
        self._state_hash = None  # Сброс хэша состояния

    @property
    def components_list(self):
        """Список компонентов композита."""
        return self.components

    @components_list.setter
    def components_list(self, comps):
        """Замена списка компонентов."""
        comps = list(comps)
        if len(comps) == 0:
            raise ValueError("components list cannot be empty")
        for c in comps:
            if not isinstance(c, DispersionABS):
                raise TypeError("All components must be instances of DispersionABS or its subclasses.")
        self.components = comps
        # Обновляем диапазон
        self.lambda_min = max(comp.lambda_min for comp in self.components)
        self.lambda_max = min(comp.lambda_max for comp in self.components)
        self._state_hash = None  # Сброс хэша состояния

    @property
    def state_hash(self) -> int:
        """
        Хэш состояния композита.
        Зависит от:
        - state_hash каждой компоненты,
        - долей (fractions).
        Пересчитывается, если:
        - изменился хэш долей (TrackedArray._hash is None),
        - или _state_hash сброшен.
        """
        if self._state_hash is None or self._fractions._hash is None:
            comp_hashes = tuple(
                comp.state_hash if hasattr(comp, 'state_hash') else id(comp)
                for comp in self.components
            )
            self._state_hash = hash((
                comp_hashes,
                self._fractions.hash,  # Пересчитает хэш долей, если он сброшен
            ))
        return self._state_hash

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




class ScaledDispersion(DispersionABS):
    """
    Материал с масштабированным комплексным показателем преломления.
    CRI_scaled(λ) = index_scaling * CRI_base(λ)
    """

    def __init__(self, base_material: DispersionABS, index_scaling: complex = 1.0, name: str = None):
        if not isinstance(base_material, DispersionABS):
            raise TypeError("base_material must be an instance of DispersionABS or its subclass.")
        super().__init__(name=name)
        
        self._base_material = base_material
        self._index_scaling = complex(index_scaling)
        
        # Наследуем диапазон от базового материала
        self.lambda_min = base_material.lambda_min
        self.lambda_max = base_material.lambda_max
        
        self._state_hash = None

    @property
    def base_material(self):
        """Базовый материал."""
        return self._base_material

    @base_material.setter
    def base_material(self, material):
        """Замена базового материала."""
        if not isinstance(material, DispersionABS):
            raise TypeError("base_material must be an instance of DispersionABS or its subclass.")
        self._base_material = material
        # Обновляем диапазон
        self.lambda_min = material.lambda_min
        self.lambda_max = material.lambda_max
        self._state_hash = None  # Сброс хэша состояния

    @property
    def index_scaling(self):
        """Комплексный коэффициент масштабирования."""
        return self._index_scaling

    @index_scaling.setter
    def index_scaling(self, val):
        """Изменение коэффициента масштабирования."""
        self._index_scaling = complex(val)
        self._state_hash = None  # Сброс хэша состояния

    @property
    def state_hash(self) -> int:
        """
        Хэш состояния масштабированного материала.
        Зависит от:
        - state_hash базового материала,
        - index_scaling.
        Пересчитывается при изменении базового материала или коэффициента.
        """
        if self._state_hash is None:
            # Хэш базового материала
            base_hash = (self._base_material.state_hash 
                        if hasattr(self._base_material, 'state_hash') 
                        else id(self._base_material))
            # Хэш состояния = (хэш_базы, коэффициент)
            self._state_hash = hash((base_hash, self._index_scaling))
        return self._state_hash

    def CRI(self, wavelength: float) -> complex:
        """
        Возвращает масштабированный комплексный показатель преломления.
        """
        return self._index_scaling * self._base_material.CRI(wavelength)

    def __repr__(self) -> str:
        base_repr = repr(self._base_material)
        scaling_str = f"{self._index_scaling:.3g}"
        name_str = f" ({self.name})" if self.name else ""
        return f"ScaledDispersion{name_str}: {scaling_str} × {base_repr}"



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
               ylim=[0, 1.05], noise=None, linestyle=None, linewidth=None):
    """
    Построение графиков с поддержкой множественных кривых, стилей и толщин линий.
    
    Parameters:
    -----------
    x : array(float) or list of arrays(float)
        x координаты. Если одно значение (array) - используется для всех кривых y.
        Если список arrays - должно соответствовать количеству y.
    y : array(float) or list of arrays(float)
        y координаты. Может быть одним массивом или списком массивов.
    title : string, default='Reflection data'
        Заголовок графика.
    dpi : int, optional
        DPI для фигуры. Если None - используется значение по умолчанию.
    tir_ang : float or list of floats, optional
        Угол(ы) полного внутреннего отражения. Отображается вертикальной пунктирной линией.
    label : string or list of strings, optional
        Подпись(и) для кривых. Если одно значение - используется для всех.
        Если список - применяется по порядку.
    ox_label : string, default='ϴ, °'
        Подпись оси X.
    oy_label : string, default='R'
        Подпись оси Y.
    ylim : list of floats, default=[0, 1.05]
        Пределы оси Y в формате [y_min, y_max].
    noise : array(float) or list of arrays(float), optional
        Шум для заливки между кривыми (fill_between).
        Если одно значение - используется для всех кривых.
    linestyle : string or list of strings, optional
        Стиль линий. Примеры: '-' (сплошная), '--' (пунктирная), '-.' (штрих-точка), ':' (точки).
        Если одно значение - используется для всех кривых.
        Если список - применяется по порядку.
    linewidth : float or list of floats, optional
        Толщина линий. Если одно значение - для всех кривых.
        Если список - применяется по порядку.
    
    Returns:
    --------
    None (отображает график через plt.show())
    
    Examples:
    ---------
    # Одиночный x для нескольких y с разными стилями
    >>> plot_graph(x_data, [y1, y2, y3],
    ...            label=['Curve 1', 'Curve 2', 'Curve 3'],
    ...            linestyle=['--', '-', '-.'],
    ...            linewidth=[1.5, 2.0, 1.0])
    
    # Несколько x и y с одиночным стилем
    >>> plot_graph([x1, x2], [y1, y2],
    ...            label=['Data 1', 'Data 2'],
    ...            linestyle='--',
    ...            linewidth=2.0)
    """
    
    # ========== БЛОК 1: ПРЕОБРАЗОВАНИЕ Y ==========
    # Определяем, является ли y одним массивом или списком массивов
    # isinstance(y[0], ...) проверяет первый элемент - если это array/list, то y - список массивов
    if isinstance(y, (list, np.ndarray)) and len(y) > 0 and isinstance(y[0], (list, np.ndarray, np.generic)):
        y_list = [np.array(yi) for yi in y]
    else:
        # y - одиночный массив, преобразуем в список с одним элементом
        y_list = [np.array(y)]
    
    num_plots = len(y_list)
    
    # ========== БЛОК 2: ПРЕОБРАЗОВАНИЕ X ==========
    # Критическая логика: определяем, одиночный это x или множественный
    is_x_single = False
    
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            raise ValueError("x не может быть пустым")
        
        # КЛЮЧЕВОЙ МОМЕНТ: проверяем первый элемент x
        # Если x[0] - это array/list, то x - это список массивов
        # Если x[0] - это число (np.generic), то x - одиночный массив
        if isinstance(x[0], (list, np.ndarray)):
            # Несколько массивов x
            x_list = [np.array(xi) for xi in x]
        else:
            # Одиночный массив x - используем его для всех y
            x_list = [np.array(x)] * num_plots
            is_x_single = True
    else:
        # x передан как скалярное значение (редко, но на случай)
        x_list = [np.array(x)] * num_plots
        is_x_single = True
    
    # Проверка соответствия количества x_list и y_list
    if not is_x_single and len(x_list) != num_plots:
        print(f"Внимание: количество массивов x ({len(x_list)}) не совпадает с количеством массивов y ({num_plots}). "
              f"Будут построены графики для первых {min(len(x_list), num_plots)} пар.")
        num_plots = min(len(x_list), num_plots)
        y_list = y_list[:num_plots]
    
    # ========== БЛОК 3: ПРЕОБРАЗОВАНИЕ LABEL ==========
    # Поддержка: одиночное значение (для всех) или список (по порядку)
    if label is None:
        labels = [None] * num_plots
    elif isinstance(label, (list, np.ndarray)):
        labels = list(label)
    else:
        # Одиночное значение - применяем ко всем графикам
        labels = [label] * num_plots
    
    # Расширяем список labels, если недостаточно значений
    if len(labels) < num_plots:
        labels.extend([None] * (num_plots - len(labels)))
    
    # ========== БЛОК 4: ПРЕОБРАЗОВАНИЕ LINESTYLE ==========
    # Поддержка: None (по умолчанию '-'), одиночное значение или список
    if linestyle is None:
        linestyles = ['-'] * num_plots
    elif isinstance(linestyle, (list, np.ndarray)):
        linestyles = list(linestyle)
    else:
        # Одиночное значение - применяем ко всем графикам
        linestyles = [linestyle] * num_plots
    
    # Расширяем список linestyles до количества графиков
    if len(linestyles) < num_plots:
        linestyles.extend(['-'] * (num_plots - len(linestyles)))
    
    # ========== БЛОК 5: ПРЕОБРАЗОВАНИЕ LINEWIDTH ==========
    # Поддержка: None (по умолчанию 1.0), одиночное значение или список
    if linewidth is None:
        linewidths = [1.5] * num_plots
    elif isinstance(linewidth, (list, np.ndarray)):
        linewidths = list(linewidth)
    else:
        # Одиночное значение - применяем ко всем графикам
        linewidths = [linewidth] * num_plots
    
    # Расширяем список linewidths до количества графиков
    if len(linewidths) < num_plots:
        linewidths.extend([1.0] * (num_plots - len(linewidths)))
    
    # ========== БЛОК 6: СОЗДАНИЕ ФИГУРЫ ==========
    if dpi is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(dpi=dpi)
    
    # Включаем сетку
    ax.grid()
    
    # ========== БЛОК 7: ПОСТРОЕНИЕ ГРАФИКОВ ==========
    # Основной цикл: строим каждую кривую с её собственными параметрами
    for i in range(num_plots):
        ax.plot(
            x_list[i],                                    # x координаты (может быть одиночный для всех)
            y_list[i],                                    # y координаты (разные для каждого)
            label=labels[i] if i < len(labels) else None,                      # подпись кривой
            linestyle=linestyles[i] if i < len(linestyles) else '-',          # стиль линии
            linewidth=linewidths[i] if i < len(linewidths) else 1.0           # толщина линии
        )
    
    # ========== БЛОК 8: ЛЕГЕНДА ==========
    # Отображаем легенду только если есть хотя бы одна подпись (не None)
    if any(labels):
        plt.legend(loc='best')
    
    # ========== БЛОК 9: ВЕРТИКАЛЬНЫЕ ЛИНИИ (TIR) ==========
    # Линии полного внутреннего отражения
    if tir_ang is not None:
        if isinstance(tir_ang, (list, np.ndarray)):
            # Несколько углов - рисуем линию для каждого
            for ang in tir_ang:
                plt.axvline(ang, linestyle="--", color='gray', alpha=0.7)
        else:
            # Одиночный угол
            plt.axvline(tir_ang, linestyle="--", color='gray', alpha=0.7)
    
    # ========== БЛОК 10: ОФОРМЛЕНИЕ ==========
    plt.title(title)
    plt.ylabel(oy_label)
    plt.xlabel(ox_label)
    
    # ========== БЛОК 11: ПРЕДЕЛЫ ОСИ Y ==========
    if ylim:
        plt.ylim(ylim)
    
    # ========== БЛОК 12: ШУМ (FILL_BETWEEN) ==========
    # Заливка между кривыми для визуализации неопределённости
    if noise is not None:
        # Преобразуем noise в список массивов (аналогично x и y)
        if isinstance(noise, (list, np.ndarray)) and len(noise) > 0 and isinstance(noise[0], (list, np.ndarray)):
            # Несколько массивов noise
            noise_list = [np.array(noisei) for noisei in noise]
        else:
            # Одиночный массив noise - используем для всех кривых
            noise_list = [np.array(noise)] * num_plots
        
        # Применяем fill_between для каждой кривой
        for i in range(min(num_plots, len(noise_list))):
            plt.fill_between(
                x_list[i],                                           # x координаты
                y_list[i] - noise_list[i],                          # нижняя граница
                y_list[i] + noise_list[i],                          # верхняя граница
                color='gray',
                alpha=0.2,
                label='±Noise' if i == 0 else ''  # подпись только для первого
            )
    
    # ========== БЛОК 13: ОТОБРАЖЕНИЕ ==========
    plt.show()


def plot_spectrum(wavelength, reflection, labels=None, title='Reflection data', dpi=None, fix_wl=None):
    """
    Plot spectral curves with wavelength-based coloring and distinct marker shapes for curve identification.
    Wavelength input is in meters.
    
    Marker shapes cycle through: 'o' (circle), 's' (square), '^' (triangle), 'D' (diamond).
    Legend shows both color (physics) and marker (identity).
    """

    # === 1. Преобразование длины волны (нм) → RGB ===
    def wl_to_rgb(w_nm):
        if w_nm < 380 or w_nm > 750:
            return (0.5, 0.5, 0.5)
        gamma = 0.8
        if w_nm < 440:
            R, G, B = -(w_nm - 440.) / (440. - 380.), 0.0, 1.0
        elif w_nm < 490:
            R, G, B = 0.0, (w_nm - 440.) / (490. - 440.), 1.0
        elif w_nm < 510:
            R, G, B = 0.0, 1.0, -(w_nm - 510.) / (510. - 490.)
        elif w_nm < 580:
            R, G, B = (w_nm - 510.) / (580. - 510.), 1.0, 0.0
        elif w_nm < 645:
            R, G, B = 1.0, -(w_nm - 645.) / (645. - 580.), 0.0
        else:
            R, G, B = 1.0, 0.0, 0.0

        factor = (
            0.3 + 0.7 * (w_nm - 380.) / (420. - 380.) if w_nm < 420 else
            0.3 + 0.7 * (750. - w_nm) / (750. - 700.) if w_nm > 700 else
            1.0
        )
        return ((R * factor) ** gamma, (G * factor) ** gamma, (B * factor) ** gamma)

    # === 2. Нормализация входных данных ===
    def to_array_list(x):
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return []
            if isinstance(x[0], (list, tuple, np.ndarray)):
                return [np.asarray(arr, dtype=float) for arr in x]
            else:
                return [np.asarray(x, dtype=float)]
        else:
            return [np.asarray(x, dtype=float)]

    wl_list = to_array_list(wavelength)
    refl_list = to_array_list(reflection)

    if len(wl_list) == 1 and len(refl_list) > 1:
        wl_list = wl_list * len(refl_list)
    elif len(refl_list) == 1 and len(wl_list) > 1:
        refl_list = refl_list * len(wl_list)
    elif len(wl_list) != len(refl_list):
        raise ValueError(f"Cannot pair {len(wl_list)} wavelength arrays with {len(refl_list)} reflection arrays.")

    n_curves = len(wl_list)

    for i, (wl, r) in enumerate(zip(wl_list, refl_list)):
        if wl.shape != r.shape:
            raise ValueError(f"Shape mismatch in curve {i}: {wl.shape} vs {r.shape}")

    # === 3. Обработка labels ===
    if labels is None:
        label_list = [None] * n_curves
    elif isinstance(labels, str):
        label_list = [labels] + [None] * (n_curves - 1)
    elif isinstance(labels, (list, tuple)):
        if len(labels) == 0:
            label_list = [None] * n_curves
        elif len(labels) == 1:
            label_list = [labels[0]] + [None] * (n_curves - 1)
        elif len(labels) == n_curves:
            label_list = list(labels)
        else:
            raise ValueError(f"Number of labels ({len(labels)}) must be 1 or {n_curves}.")
    else:
        raise TypeError("labels must be None, str, or list/tuple.")

    # === 4. Определение форм маркеров ===
    marker_cycle = ['o',  '^', 's','D']  # круг, квадрат, треугольник, ромб
    markers = [marker_cycle[i % len(marker_cycle)] for i in range(n_curves)]

    # === 5. Создание холста ===
    fig, ax = plt.subplots(dpi=dpi) if dpi else plt.subplots()

    # === 6. Отрисовка кривых ===
    scatter_handles = []
    any_label = False

    for i, (wl, r, lbl, mk) in enumerate(zip(wl_list, refl_list, label_list, markers)):
        wl_nm = wl * 1e9
        colors = [wl_to_rgb(w) for w in wl_nm]
        sc = ax.scatter(wl_nm, r, c=colors, s=25, marker=mk, edgecolors='none', label=lbl)
        if lbl is not None:
            any_label = True
            scatter_handles.append(sc)

    # === 7. Легенда с правильными маркерами ===
    if any_label:
        # Используем стандартную легенду — matplotlib сам подхватит маркер из scatter
        ax.legend(loc='best')

    # === 8. Завершение оформления ===
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('λ, нм')
    ax.set_ylabel('R')
    ax.set_ylim(0, 1.05)

    # Вертикальные линии (если есть)
    if fix_wl is not None:
        tir_list = [fix_wl] if not isinstance(fix_wl, (list, tuple, np.ndarray)) else fix_wl
        for ang in tir_list:
            ax.axvline(ang * 1e9, color='gray', linestyle='--', linewidth=1)

    plt.show()  


def plot_complex_reflection(r, ang=None, title='', fix_limits=True, cbar_shrink=0.75, cbar_pad=0.05):
    """
    Строит комплексные коэффициенты отражения на комплексной плоскости
    
    Параметры:
    r : array-like
        Массив комплексных коэффициентов отражения
    ang : array-like, optional
        Массив углов падения для цветовой кодировки
    title : str, optional
        Заголовок графика
    fix_limits : bool, optional
        Фиксировать ли пределы осей
    cbar_shrink : float, optional
        Уменьшение высоты colorbar (1.0 = полная высота)
    cbar_pad : float, optional
        Отступ colorbar от графика
    """
    # Увеличиваем ширину фигуры для colorbar
    fig, ax = plt.subplots(figsize=(11, 10))  # Было (10, 10) -> 11 по ширине
    
    # Единичная окружность
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', 
                       alpha=0.7, linewidth=1.5)
    ax.add_artist(circle)
    
    # Отрисовка траектории
    if ang is not None:
        sc = ax.scatter(r.real, r.imag, c=ang, cmap='viridis', s=40,
                       edgecolor='k', alpha=0.9, zorder=10)
        ax.plot(r.real, r.imag, 'b-', alpha=0.6, linewidth=2, zorder=5)
        
        # Ключевое исправление: уменьшаем высоту colorbar и увеличиваем отступ
        cbar = fig.colorbar(sc, ax=ax, shrink=cbar_shrink, pad=cbar_pad)
        cbar.set_label('Угол падения (°)', fontsize=12, fontweight='bold')
        
        # Уменьшаем шрифт меток на colorbar для компактности
        cbar.ax.tick_params(labelsize=10)
    else:
        ax.scatter(r.real, r.imag, s=40, color='blue', edgecolor='k',
                  alpha=0.9, zorder=10)
        ax.plot(r.real, r.imag, 'b-', alpha=0.6, linewidth=2, zorder=5)
    
    # Подписи ключевых точек
    ax.text(r.real[0], r.imag[0], ' 0°', fontsize=10, fontweight='bold',
           ha='left', va='bottom')
    ax.text(r.real[-1], r.imag[-1], ' 90°', fontsize=10, fontweight='bold',
           ha='right', va='top')
    
    # Оси и сетка
    ax.axhline(0, color='black', linewidth=1.0, alpha=0.8)
    ax.axvline(0, color='black', linewidth=1.0, alpha=0.8)
    ax.set_xlabel('Re($r_p$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Im($r_p$)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    
    if fix_limits:
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    
    # Критически важные настройки после tight_layout
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)  # Оставляем место для colorbar
    plt.show()


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



# -------------------------------------------------------------------------
# ---------------- Вспомогаетльные структуры ------------------------------
# -------------------------------------------------------------------------


class TrackedArray:
    """
    Минималистичная обёртка над массивом numpy, отслеживающая изменения через сброс хэша.
    """

    def __init__(self, data):
        """
        Инициализация TrackedArray.

        Args:
            data: Входные данные (скаляр или массив).
        """
        self._data = data
        self._hash = None  # Хэш сбрасывается при изменении

    def __getitem__(self, index):
        """Возвращает элемент массива по индексу или срезу."""
        return self._data[index]

    def __setitem__(self, index, value):
        """Устанавливает значение элемента массива по индексу или срезу."""
        self._data[index] = value
        self._hash = None  # Сброс хэша при изменении элемента

    def __len__(self):
        """Возвращает длину массива."""
        return len(self._data)

    def __array__(self):
        """Возвращает внутренний массив numpy."""
        return self._data

    @property
    def hash(self):
        """Возвращает хэш массива. Если хэш сброшен, пересчитывает его."""
        if self._hash is None:
            self._hash = hash(tuple(self._data))
        return self._hash

