from pathlib import Path
import re

ROOT = Path.cwd()

def read(p):
    path = ROOT / p
    return path.read_text(encoding='utf-8', errors='replace')

def write(p, s):
    path = ROOT / p
    path.write_text(s, encoding='utf-8', newline='\n')
    print(f'patched {p}')

# -----------------------------
# fibersim/treePO.cu
# -----------------------------
p = 'fibersim/treePO.cu'
s = read(p)

# Replace the fragile include block.  This intentionally matches from stdio.h
# through the last standard include before project-local util.h.
include_re = re.compile(r'''#include\s*<stdio\.h>\s*\n#include\s*<iostream>\s*\n#include\s*<fstream>\s*\n#include\s*<sys/stat\.h>\s*\n(?:#ifdef\s+_WIN32.*?#endif\s*\n|#include\s*<sys/time\.h>\s*\n)?#include\s*<cufft\.h>\s*\n#include\s*<cufftXt\.h>\s*\n(?:\s*//\s*)?#include\s*<bits/stdc\+\+\.h>\s*\n''', re.S)
new_includes = r'''#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <complex>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <utility>
#include <limits>
#include <random>
#include <chrono>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
inline int gettimeofday(timeval* tp, void*) {
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static const unsigned __int64 EPOCH = 116444736000000000ULL;
    if (tp) {
        GetSystemTimeAsFileTime(&ft);
        tmpres |= ((unsigned __int64)ft.dwHighDateTime) << 32;
        tmpres |= ft.dwLowDateTime;
        tmpres -= EPOCH;
        tp->tv_sec  = (long)(tmpres / 10000000ULL);
        tp->tv_usec = (long)((tmpres % 10000000ULL) / 10ULL);
    }
    return 0;
}
#else
#include <sys/stat.h>
#include <sys/time.h>
#endif

#include <cufft.h>
#include <cufftXt.h>
'''
if include_re.search(s):
    s = include_re.sub(new_includes, s, count=1)
else:
    # Less strict fallback: remove bits and sys/time include; insert missing includes after fstream.
    s = s.replace('#include <sys/time.h>\n', '')
    s = s.replace('#include <bits/stdc++.h>\n', '')
    s = s.replace('// #include <bits/stdc++.h>\n', '')
    s = s.replace('#include <fstream>\n', '#include <fstream>\n#include <cstdlib>\n#include <vector>\n#include <string>\n#include <sstream>\n#include <iomanip>\n#include <algorithm>\n#include <numeric>\n#include <cmath>\n#include <complex>\n#include <map>\n#include <set>\n#include <unordered_map>\n#include <unordered_set>\n#include <tuple>\n#include <utility>\n#include <limits>\n#include <random>\n#include <chrono>\n#include <filesystem>\n')
    if 'inline int gettimeofday(timeval* tp' not in s:
        s = s.replace('#include <sys/stat.h>\n', '#include <sys/stat.h>\n#ifdef _WIN32\n#ifndef NOMINMAX\n#define NOMINMAX\n#endif\n#include <winsock2.h>\n#include <windows.h>\ninline int gettimeofday(timeval* tp, void*) {\n    FILETIME ft;\n    unsigned __int64 tmpres = 0;\n    static const unsigned __int64 EPOCH = 116444736000000000ULL;\n    if (tp) {\n        GetSystemTimeAsFileTime(&ft);\n        tmpres |= ((unsigned __int64)ft.dwHighDateTime) << 32;\n        tmpres |= ft.dwLowDateTime;\n        tmpres -= EPOCH;\n        tp->tv_sec  = (long)(tmpres / 10000000ULL);\n        tp->tv_usec = (long)((tmpres % 10000000ULL) / 10ULL);\n    }\n    return 0;\n}\n#else\n#include <sys/time.h>\n#endif\n')

# POSIX mkdir(path, mode) -> portable C++17 filesystem.
s = re.sub(r'\bmkdir\s*\(\s*outputdir\.c_str\s*\(\s*\)\s*,\s*0777\s*\)\s*;',
           'std::filesystem::create_directories(outputdir);', s)

# Variable length arrays used by MSVC-host compilation.
s = re.sub(r'int\s+thetaonumvec\s*\[\s*max_depth\s*\+\s*1\s*\]\s*;',
           'std::vector<int> thetaonumvec(max_depth + 1);', s)
s = re.sub(r'int\s+phionumvec\s*\[\s*max_depth\s*\+\s*1\s*\]\s*;',
           'std::vector<int> phionumvec(max_depth + 1);', s)
s = re.sub(r'int\s+anglenumvec\s*\[\s*max_depth\s*\+\s*1\s*\]\s*;',
           'std::vector<int> anglenumvec(max_depth + 1);', s)

write(p, s)

# -----------------------------
# util/util.h
# -----------------------------
p = 'util/util.h'
s = read(p)

# Provide M_PI fallback; this also fixes any remaining uses instead of only mu0.
if '#ifndef M_PI' not in s:
    # Insert after helper_cuda include if possible.
    s = re.sub(r'(#include\s+"helper_cuda\.h"\s*)', r'\1\n#ifndef M_PI\n#define M_PI 3.14159265358979323846\n#endif\n', s, count=1)

# If mu0 already uses M_PI, make it float-safe.
s = re.sub(r'float\s+mu0\s*=\s*4e-7\s*\*\s*M_PI\s*;',
           'float mu0 = 4e-7f * static_cast<float>(M_PI);', s)

# CUDA 13 / CCCL no longer has thrust::binary_function, and float6 is a tuple.
add6_re = re.compile(r'''struct\s+add6\s*(?::\s*public\s+thrust::binary_function\s*<\s*float6\s*,\s*float6\s*,\s*float6\s*>\s*)?\s*\{\s*__host__\s+__device__\s+float6\s+operator\s*\(\s*\)\s*\(\s*const\s+float6\s*&\s*a\s*,\s*const\s+float6\s*&\s*b\s*\)\s*const\s*\{.*?return\s+c\s*;\s*\}\s*\}\s*;''', re.S)
new_add6 = r'''struct add6 {
  __host__ __device__
  float6 operator()(const float6& a, const float6& b) const {
    float6 c;
    thrust::get<0>(c) = thrust::get<0>(a) + thrust::get<0>(b);
    thrust::get<1>(c) = thrust::get<1>(a) + thrust::get<1>(b);
    thrust::get<2>(c) = thrust::get<2>(a) + thrust::get<2>(b);
    thrust::get<3>(c) = thrust::get<3>(a) + thrust::get<3>(b);
    thrust::get<4>(c) = thrust::get<4>(a) + thrust::get<4>(b);
    thrust::get<5>(c) = thrust::get<5>(a) + thrust::get<5>(b);
    return c;
  }
};'''
if add6_re.search(s):
    s = add6_re.sub(new_add6, s, count=1)
else:
    # Simpler fallback for the current broken add6 body that uses .x/.y/.z/.u/.v/.w.
    s = re.sub(r'struct\s+add6\s*\{\s*__host__\s+__device__\s+float6\s+operator\(\)\(const float6& a, const float6& b\) const \{\s*float6 c;\s*c\.x = a\.x \+ b\.x;\s*c\.y = a\.y \+ b\.y;\s*c\.z = a\.z \+ b\.z;\s*c\.u = a\.u \+ b\.u;\s*c\.v = a\.v \+ b\.v;\s*c\.w = a\.w \+ b\.w;\s*return c;\s*\}\s*\};', new_add6, s, flags=re.S)

write(p, s)

# -----------------------------
# util/helper_cuda.h
# -----------------------------
p = 'util/helper_cuda.h'
s = read(p)
# Guard the three cuFFT enum cases removed in CUDA 13.
# This is intentionally conservative: it wraps each case block until the next case label.
for enum_name in ['CUFFT_INCOMPLETE_PARAMETER_LIST', 'CUFFT_PARSE_ERROR', 'CUFFT_LICENSE_ERROR']:
    if enum_name in s and f'#if CUDART_VERSION < 13000\n    case {enum_name}' not in s:
        pattern = re.compile(r'(\s*case\s+' + enum_name + r'\s*:\s*\n\s*return\s+"' + enum_name + r'"\s*;)', re.S)
        s = pattern.sub(r'\n#if CUDART_VERSION < 13000\1\n#endif', s)

write(p, s)

# -----------------------------
# util/fields.h
# -----------------------------
p = 'util/fields.h'
s = read(p)
# MSVC does not support VLAs.
s = re.sub(r'float\s+scattering\s*\[\s*thetaonum\s*\*\s*phiofinal\s*\]\s*=\s*\{\s*0\.f\s*\}\s*;',
           'std::vector<float> scattering(thetaonum * phiofinal, 0.0f);', s)
write(p, s)

print('\nDone. Now try:')
print('  cd fibersim')
print('  nvcc -std=c++17 treePO.cu -rdc=true -o treePO -lcufft')
