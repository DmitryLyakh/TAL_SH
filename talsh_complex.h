/** ExaTensor::TAL-SH: Complex arithmetic header.
REVISION: 2017/05/17

Copyright (C) 2014-2017 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2017 Oak Ridge National Laboratory (UT-Battelle)

This file is part of ExaTensor.

ExaTensor is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ExaTensor is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
**/

#ifndef _TALSH_COMPLEX_H
#define _TALSH_COMPLEX_H

#ifdef __cplusplus
#include <complex>
#endif

#ifndef NO_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

//DECLARATIONS:
// Complex number:
#ifndef NO_GPU
typedef cuFloatComplex talshComplex4;
typedef cuDoubleComplex talshComplex8;
#else
#ifdef __cplusplus
typedef std::complex<float> talshComplex4;
typedef std::complex<double> talshComplex8;
#else
typedef struct{float real; float imag;} talshComplex4;
typedef struct{double real; double imag;} talshComplex8;
#endif
#endif /*NO_GPU*/

/* TAL-SH complex arithmetic headers:
inline talshComplex4 talshComplex4Set(float real, float imag);
inline talshComplex8 talshComplex8Set(double real, double imag);
inline float talshComplex4Real(talshComplex4 cmplx);
inline double talshComplex8Real(talshComplex8 cmplx);
inline float talshComplex4Imag(talshComplex4 cmplx);
inline double talshComplex8Imag(talshComplex8 cmplx);
inline talshComplex4 talshComplex4Conjg(talshComplex4 cmplx);
inline talshComplex8 talshComplex8Conjg(talshComplex8 cmplx);
inline float talshComplex4Abs(talshComplex4 cmplx);
inline double talshComplex8Abs(talshComplex8 cmplx);
*/

//DEFINITIONS:
// Complex arithmetic:
inline talshComplex4 talshComplex4Set(float real, float imag)
{
#ifndef NO_GPU
 talshComplex4 result = make_cuFloatComplex(real,imag);
#else
#ifdef __cplusplus
 talshComplex4 result(real,imag);
#else
 talshComplex4 result = {real,imag};
#endif
#endif
 return result;
}

inline talshComplex8 talshComplex8Set(double real, double imag)
{
#ifndef NO_GPU
 talshComplex8 result = make_cuDoubleComplex(real,imag);
#else
#ifdef __cplusplus
 talshComplex8 result(real,imag);
#else
 talshComplex8 result = {real,imag};
#endif
#endif
 return result;
}

inline float talshComplex4Real(talshComplex4 cmplx)
{
#ifndef NO_GPU
 return cuCrealf(cmplx);
#else
#ifdef __cplusplus
 return cmplx.real();
#else
 return cmplx.real;
#endif
#endif
}

inline double talshComplex8Real(talshComplex8 cmplx)
{
#ifndef NO_GPU
 return cuCreal(cmplx);
#else
#ifdef __cplusplus
 return cmplx.real();
#else
 return cmplx.real;
#endif
#endif
}

inline float talshComplex4Imag(talshComplex4 cmplx)
{
#ifndef NO_GPU
 return cuCimagf(cmplx);
#else
#ifdef __cplusplus
 return cmplx.imag();
#else
 return cmplx.imag;
#endif
#endif
}

inline double talshComplex8Imag(talshComplex8 cmplx)
{
#ifndef NO_GPU
 return cuCimag(cmplx);
#else
#ifdef __cplusplus
 return cmplx.imag();
#else
 return cmplx.imag;
#endif
#endif
}

inline talshComplex4 talshComplex4Conjg(talshComplex4 cmplx)
{
#ifndef NO_GPU
 return cuConjf(cmplx);
#else
#ifdef __cplusplus
 return std::conj(cmplx);
#else
 talshComplex4 result = {cmplx.real,-cmplx.imag};
 return result;
#endif
#endif
}

inline talshComplex8 talshComplex8Conjg(talshComplex8 cmplx)
{
#ifndef NO_GPU
 return cuConj(cmplx);
#else
#ifdef __cplusplus
 return std::conj(cmplx);
#else
 talshComplex8 result = {cmplx.real,-cmplx.imag};
 return result;
#endif
#endif
}

inline float talshComplex4Abs(talshComplex4 cmplx)
{
#ifndef NO_GPU
 return cuCabsf(cmplx);
#else
#ifdef __cplusplus
 return std::abs(cmplx);
#else
 return (float)sqrt((double)((cmplx.real)*(cmplx.real)) + (double)((cmplx.imag)*(cmplx.imag)));
#endif
#endif
}

inline double talshComplex8Abs(talshComplex8 cmplx)
{
#ifndef NO_GPU
 return cuCabs(cmplx);
#else
#ifdef __cplusplus
 return std::abs(cmplx);
#else
 return sqrt(((cmplx.real)*(cmplx.real)) + ((cmplx.imag)*(cmplx.imag)));
#endif
#endif
}

#endif /*_TALSH_COMPLEX_H*/
