#
# The ndarray object from _testbuffer.c is a complete implementation of
# a PEP-3118 buffer provider. It is independent from NumPy's ndarray
# and the tests don't require NumPy.
#
# If NumPy is present, some tests check both ndarray implementations
# against each other.
#
# Most ndarray tests also check that memoryview(ndarray) behaves in
# the same way as the original. Thus, a substantial part of the
# memoryview tests is now in this module.
#
# Written and designed by Stefan Krah for Python 3.3.
#

import contextlib
import unittest
from test import support
from itertools import permutations, product
from random import randrange, sample, choice
import warnings
import sys, array, io, os
from decimal import Decimal
from fractions import Fraction

try:
    from _testbuffer import *
except ImportError:
    ndarray = None

try:
    import struct
except ImportError:
    struct = None

try:
    import ctypes
except ImportError:
    ctypes = None

try:
    with support.EnvironmentVarGuard() as os.environ, \
         warnings.catch_warnings():
        from numpy import ndarray as numpy_array
except ImportError:
    numpy_array = None

try:
    import _testcapi
except ImportError:
    _testcapi = None


SHORT_TEST = True


# ======================================================================
#                    Random lists by format specifier
# ======================================================================

# Native format chars and their ranges.
NATIVE = {
    '?':0, 'c':0, 'b':0, 'B':0,
    'h':0, 'H':0, 'i':0, 'I':0,
    'l':0, 'L':0, 'n':0, 'N':0,
    'f':0, 'd':0, 'P':0
}

# NumPy does not have 'n' or 'N':
if numpy_array:
    del NATIVE['n']
    del NATIVE['N']

if struct:
    try:
        # Add "qQ" if present in native mode.
        struct.pack('Q', 2**64-1)
        NATIVE['q'] = 0
        NATIVE['Q'] = 0
    except struct.error:
        pass

# Standard format chars and their ranges.
STANDARD = {
    '?':(0, 2),            'c':(0, 1<<8),
    'b':(-(1<<7), 1<<7),   'B':(0, 1<<8),
    'h':(-(1<<15), 1<<15), 'H':(0, 1<<16),
    'i':(-(1<<31), 1<<31), 'I':(0, 1<<32),
    'l':(-(1<<31), 1<<31), 'L':(0, 1<<32),
    'q':(-(1<<63), 1<<63), 'Q':(0, 1<<64),
    'f':(-(1<<63), 1<<63), 'd':(-(1<<1023), 1<<1023)
}

def native_type_range(fmt):
    """Return range of a native type."""
    if fmt == 'c':
        lh = (0, 256)
    elif fmt == '?':
        lh = (0, 2)
    elif fmt == 'f':
        lh = (-(1<<63), 1<<63)
    elif fmt == 'd':
        lh = (-(1<<1023), 1<<1023)
    else:
        for exp in (128, 127, 64, 63, 32, 31, 16, 15, 8, 7):
            try:
                struct.pack(fmt, (1<<exp)-1)
                break
            except struct.error:
                pass
        lh = (-(1<<exp), 1<<exp) if exp & 1 else (0, 1<<exp)
    return lh

fmtdict = {
    '':NATIVE,
    '@':NATIVE,
    '<':STANDARD,
    '>':STANDARD,
    '=':STANDARD,
    '!':STANDARD
}

if struct:
    for fmt in fmtdict['@']:
        fmtdict['@'][fmt] = native_type_range(fmt)

MEMORYVIEW = NATIVE.copy()
ARRAY = NATIVE.copy()
for k in NATIVE:
    if not k in "bBhHiIlLfd":
        del ARRAY[k]

BYTEFMT = NATIVE.copy()
for k in NATIVE:
    if not k in "Bbc":
        del BYTEFMT[k]

fmtdict['m']  = MEMORYVIEW
fmtdict['@m'] = MEMORYVIEW
fmtdict['a']  = ARRAY
fmtdict['b']  = BYTEFMT
fmtdict['@b']  = BYTEFMT

# Capabilities of the test objects:
MODE = 0
MULT = 1
cap = {         # format chars                  # multiplier
  'ndarray':    (['', '@', '<', '>', '=', '!'], ['', '1', '2', '3']),
  'array':      (['a'],                         ['']),
  'numpy':      ([''],                          ['']),
  'memoryview': (['@m', 'm'],                   ['']),
  'bytefmt':    (['@b', 'b'],                   ['']),
}

def randrange_fmt(mode, char, obj):
    """Return random item for a type specified by a mode and a single
       format character."""
    x = randrange(*fmtdict[mode][char])
    if char == 'c':
        x = bytes([x])
        if obj == 'numpy' and x == b'\x00':
            # http://projects.scipy.org/numpy/ticket/1925
            x = b'\x01'
    if char == '?':
        x = bool(x)
    if char == 'f' or char == 'd':
        x = struct.pack(char, x)
        x = struct.unpack(char, x)[0]
    return x

def gen_item(fmt, obj):
    """Return single random item."""
    mode, chars = fmt.split('#')
    x = []
    for c in chars:
        x.append(randrange_fmt(mode, c, obj))
    return x[0] if len(x) == 1 else tuple(x)

def gen_items(n, fmt, obj):
    """Return a list of random items (or a scalar)."""
    if n == 0:
        return gen_item(fmt, obj)
    lst = [0] * n
    for i in range(n):
        lst[i] = gen_item(fmt, obj)
    return lst

def struct_items(n, obj):
    mode = choice(cap[obj][MODE])
    xfmt = mode + '#'
    fmt = mode.strip('amb')
    nmemb = randrange(2, 10) # number of struct members
    for _ in range(nmemb):
        char = choice(tuple(fmtdict[mode]))
        multiplier = choice(cap[obj][MULT])
        xfmt += (char * int(multiplier if multiplier else 1))
        fmt += (multiplier + char)
    items = gen_items(n, xfmt, obj)
    item = gen_item(xfmt, obj)
    return fmt, items, item

def randitems(n, obj='ndarray', mode=None, char=None):
    """Return random format, items, item."""
    if mode is None:
        mode = choice(cap[obj][MODE])
    if char is None:
        char = choice(tuple(fmtdict[mode]))
    multiplier = choice(cap[obj][MULT])
    fmt = mode + '#' + char * int(multiplier if multiplier else 1)
    items = gen_items(n, fmt, obj)
    item = gen_item(fmt, obj)
    fmt = mode.strip('amb') + multiplier + char
    return fmt, items, item

def iter_mode(n, obj='ndarray'):
    """Iterate through supported mode/char combinations."""
    for mode in cap[obj][MODE]:
        for char in fmtdict[mode]:
            yield randitems(n, obj, mode, char)

def iter_format(nitems, testobj='ndarray'):
    """Yield (format, items, item) for all possible modes and format
       characters plus one random compound format string."""
    for t in iter_mode(nitems, testobj):
        yield t
    if testobj != 'ndarray':
        return
    yield struct_items(nitems, testobj)


def is_byte_format(fmt):
    return 'c' in fmt or 'b' in fmt or 'B' in fmt

def is_memoryview_format(fmt):
    """format suitable for memoryview"""
    x = len(fmt)
    return ((x == 1 or (x == 2 and fmt[0] == '@')) and
            fmt[x-1] in MEMORYVIEW)

NON_BYTE_FORMAT = [c for c in fmtdict['@'] if not is_byte_format(c)]


# ======================================================================
#       Multi-dimensional tolist(), slicing and slice assignments
# ======================================================================

def atomp(lst):
    """Tuple items (representing structs) are regarded as atoms."""
    return not isinstance(lst, list)

def listp(lst):
    return isinstance(lst, list)

def prod(lst):
    """Product of list elements."""
    if len(lst) == 0:
        return 0
    x = lst[0]
    for v in lst[1:]:
        x *= v
    return x

def strides_from_shape(ndim, shape, itemsize, layout):
    """Calculate strides of a contiguous array. Layout is 'C' or
       'F' (Fortran)."""
    if ndim == 0:
        return ()
    if layout == 'C':
        strides = list(shape[1:]) + [itemsize]
        for i in range(ndim-2, -1, -1):
            strides[i] *= strides[i+1]
    else:
        strides = [itemsize] + list(shape[:-1])
        for i in range(1, ndim):
            strides[i] *= strides[i-1]
    return strides

def _ca(items, s):
    """Convert flat item list to the nested list representation of a
       multidimensional C array with shape 's'."""
    if atomp(items):
        return items
    if len(s) == 0:
        return items[0]
    lst = [0] * s[0]
    stride = len(items) // s[0] if s[0] else 0
    for i in range(s[0]):
        start = i*stride
        lst[i] = _ca(items[start:start+stride], s[1:])
    return lst

def _fa(items, s):
    """Convert flat item list to the nested list representation of a
       multidimensional Fortran array with shape 's'."""
    if atomp(items):
        return items
    if len(s) == 0:
        return items[0]
    lst = [0] * s[0]
    stride = s[0]
    for i in range(s[0]):
        lst[i] = _fa(items[i::stride], s[1:])
    return lst

def carray(items, shape):
    if listp(items) and not 0 in shape and prod(shape) != len(items):
        raise ValueError("prod(shape) != len(items)")
    return _ca(items, shape)

def farray(items, shape):
    if listp(items) and not 0 in shape and prod(shape) != len(items):
        raise ValueError("prod(shape) != len(items)")
    return _fa(items, shape)

def indices(shape):
    """Generate all possible tuples of indices."""
    iterables = [range(v) for v in shape]
    return product(*iterables)

def getindex(ndim, ind, strides):
    """Convert multi-dimensional index to the position in the flat list."""
    ret = 0
    for i in range(ndim):
        ret += strides[i] * ind[i]
    return ret

def transpose(src, shape):
    """Transpose flat item list that is regarded as a multi-dimensional
       matrix defined by shape: dest...[k][j][i] = src[i][j][k]...  """
    if not shape:
        return src
    ndim = len(shape)
    sstrides = strides_from_shape(ndim, shape, 1, 'C')
    dstrides = strides_from_shape(ndim, shape[::-1], 1, 'C')
    dest = [0] * len(src)
    for ind in indices(shape):
        fr = getindex(ndim, ind, sstrides)
        to = getindex(ndim, ind[::-1], dstrides)
        dest[to] = src[fr]
    return dest

def _flatten(lst):
    """flatten list"""
    if lst == []:
        return lst
    if atomp(lst):
        return [lst]
    return _flatten(lst[0]) + _flatten(lst[1:])

def flatten(lst):
    """flatten list or return scalar"""
    if atomp(lst): # scalar
        return lst
    return _flatten(lst)

def slice_shape(lst, slices):
    """Get the shape of lst after slicing: slices is a list of slice
       objects."""
    if atomp(lst):
        return []
    return [len(lst[slices[0]])] + slice_shape(lst[0], slices[1:])

def multislice(lst, slices):
    """Multi-dimensional slicing: slices is a list of slice objects."""
    if atomp(lst):
        return lst
    return [multislice(sublst, slices[1:]) for sublst in lst[slices[0]]]

def m_assign(llst, rlst, lslices, rslices):
    """Multi-dimensional slice assignment: llst and rlst are the operands,
       lslices and rslices are lists of slice objects. llst and rlst must
       have the same structure.

       For a two-dimensional example, this is not implemented in Python:

         llst[0:3:2, 0:3:2] = rlst[1:3:1, 1:3:1]

       Instead we write:

         lslices = [slice(0,3,2), slice(0,3,2)]
         rslices = [slice(1,3,1), slice(1,3,1)]
         multislice_assign(llst, rlst, lslices, rslices)
    """
    if atomp(rlst):
        return rlst
    rlst = [m_assign(l, r, lslices[1:], rslices[1:])
            for l, r in zip(llst[lslices[0]], rlst[rslices[0]])]
    llst[lslices[0]] = rlst
    return llst

def cmp_structure(llst, rlst, lslices, rslices):
    """Compare the structure of llst[lslices] and rlst[rslices]."""
    lshape = slice_shape(llst, lslices)
    rshape = slice_shape(rlst, rslices)
    if (len(lshape) != len(rshape)):
        return -1
    for i in range(len(lshape)):
        if lshape[i] != rshape[i]:
            return -1
        if lshape[i] == 0:
            return 0
    return 0

def multislice_assign(llst, rlst, lslices, rslices):
    """Return llst after assigning: llst[lslices] = rlst[rslices]"""
    if cmp_structure(llst, rlst, lslices, rslices) < 0:
        raise ValueError("lvalue and rvalue have different structures")
    return m_assign(llst, rlst, lslices, rslices)


# ======================================================================
#                          Random structures
# ======================================================================

#
# PEP-3118 is very permissive with respect to the contents of a
# Py_buffer. In particular:
#
#   - shape can be zero
#   - strides can be any integer, including zero
#   - offset can point to any location in the underlying
#     memory block, provided that it is a multiple of
#     itemsize.
#
# The functions in this section test and verify random structures
# in full generality. A structure is valid iff it fits in the
# underlying memory block.
#
# The structure 't' (short for 'tuple') is fully defined by:
#
#   t = (memlen, itemsize, ndim, shape, strides, offset)
#

def verify_structure(memlen, itemsize, ndim, shape, strides, offset):
    """Verify that the parameters represent a valid array within
       the bounds of the allocated memory:
           char *mem: start of the physical memory block
           memlen: length of the physical memory block
           offset: (char *)buf - mem
    """
    if offset % itemsize:
        return False
    if offset < 0 or offset+itemsize > memlen:
        return False
    if any(v % itemsize for v in strides):
        return False

    if ndim <= 0:
        return ndim == 0 and not shape and not strides
    if 0 in shape:
        return True

    imin = sum(strides[j]*(shape[j]-1) for j in range(ndim)
               if strides[j] <= 0)
    imax = sum(strides[j]*(shape[j]-1) for j in range(ndim)
               if strides[j] > 0)

    return 0 <= offset+imin and offset+imax+itemsize <= memlen

def get_item(lst, indices):
    for i in indices:
        lst = lst[i]
    return lst

def memory_index(indices, t):
    """Location of an item in the underlying memory."""
    memlen, itemsize, ndim, shape, strides, offset = t
    p = offset
    for i in range(ndim):
        p += strides[i]*indices[i]
    return p

def is_overlapping(t):
    """The structure 't' is overlapping if at least one memory location
       is visited twice while iterating through all possible tuples of
       indices."""
    memlen, itemsize, ndim, shape, strides, offset = t
    visited = 1<<memlen
    for ind in indices(shape):
        i = memory_index(ind, t)
        bit = 1<<i
        if visited & bit:
            return True
        visited |= bit
    return False

def rand_structure(itemsize, valid, maxdim=5, maxshape=16, shape=()):
    """Return random structure:
           (memlen, itemsize, ndim, shape, strides, offset)
       If 'valid' is true, the returned structure is valid, otherwise invalid.
       If 'shape' is given, use that instead of creating a random shape.
    """
    if not shape:
        ndim = randrange(maxdim+1)
        if (ndim == 0):
            if valid:
                return itemsize, itemsize, ndim, (), (), 0
            else:
                nitems = randrange(1, 16+1)
                memlen = nitems * itemsize
                offset = -itemsize if randrange(2) == 0 else memlen
                return memlen, itemsize, ndim, (), (), offset

        minshape = 2
        n = randrange(100)
        if n >= 95 and valid:
            minshape = 0
        elif n >= 90:
            minshape = 1
        shape = [0] * ndim

        for i in range(ndim):
            shape[i] = randrange(minshape, maxshape+1)
    else:
        ndim = len(shape)

    maxstride = 5
    n = randrange(100)
    zero_stride = True if n >= 95 and n & 1 else False

    strides = [0] * ndim
    strides[ndim-1] = itemsize * randrange(-maxstride, maxstride+1)
    if not zero_stride and strides[ndim-1] == 0:
        strides[ndim-1] = itemsize

    for i in range(ndim-2, -1, -1):
        maxstride *= shape[i+1] if shape[i+1] else 1
        if zero_stride:
            strides[i] = itemsize * randrange(-maxstride, maxstride+1)
        else:
            strides[i] = ((1,-1)[randrange(2)] *
                          itemsize * randrange(1, maxstride+1))

    imin = imax = 0
    if not 0 in shape:
        imin = sum(strides[j]*(shape[j]-1) for j in range(ndim)
                   if strides[j] <= 0)
        imax = sum(strides[j]*(shape[j]-1) for j in range(ndim)
                   if strides[j] > 0)

    nitems = imax - imin
    if valid:
        offset = -imin * itemsize
        memlen = offset + (imax+1) * itemsize
    else:
        memlen = (-imin + imax) * itemsize
        offset = -imin-itemsize if randrange(2) == 0 else memlen
    return memlen, itemsize, ndim, shape, strides, offset

def randslice_from_slicelen(slicelen, listlen):
    """Create a random slice of len slicelen that fits into listlen."""
    maxstart = listlen - slicelen
    start = randrange(maxstart+1)
    maxstep = (listlen - start) // slicelen if slicelen else 1
    step = randrange(1, maxstep+1)
    stop = start + slicelen * step
    s = slice(start, stop, step)
    _, _, _, control = slice_indices(s, listlen)
    if control != slicelen:
        raise RuntimeError
    return s

def randslice_from_shape(ndim, shape):
    """Create two sets of slices for an array x with shape 'shape'
       such that shapeof(x[lslices]) == shapeof(x[rslices])."""
    lslices = [0] * ndim
    rslices = [0] * ndim
    for n in range(ndim):
        l = shape[n]
        slicelen = randrange(1, l+1) if l > 0 else 0
        lslices[n] = randslice_from_slicelen(slicelen, l)
        rslices[n] = randslice_from_slicelen(slicelen, l)
    return tuple(lslices), tuple(rslices)

def rand_aligned_slices(maxdim=5, maxshape=16):
    """Create (lshape, rshape, tuple(lslices), tuple(rslices)) such that
       shapeof(x[lslices]) == shapeof(y[rslices]), where x is an array
       with shape 'lshape' and y is an array with shape 'rshape'."""
    ndim = randrange(1, maxdim+1)
    minshape = 2
    n = randrange(100)
    if n >= 95:
        minshape = 0
    elif n >= 90:
        minshape = 1
    all_random = True if randrange(100) >= 80 else False
    lshape = [0]*ndim; rshape = [0]*ndim
    lslices = [0]*ndim; rslices = [0]*ndim

    for n in range(ndim):
        small = randrange(minshape, maxshape+1)
        big = randrange(minshape, maxshape+1)
        if big < small:
            big, small = small, big

        # Create a slice that fits the smaller value.
        if all_random:
            start = randrange(-small, small+1)
            stop = randrange(-small, small+1)
            step = (1,-1)[randrange(2)] * randrange(1, small+2)
            s_small = slice(start, stop, step)
            _, _, _, slicelen = slice_indices(s_small, small)
        else:
            slicelen = randrange(1, small+1) if small > 0 else 0
            s_small = randslice_from_slicelen(slicelen, small)

        # Create a slice of the same length for the bigger value.
        s_big = randslice_from_slicelen(slicelen, big)
        if randrange(2) == 0:
            rshape[n], lshape[n] = big, small
            rslices[n], lslices[n] = s_big, s_small
        else:
            rshape[n], lshape[n] = small, big
            rslices[n], lslices[n] = s_small, s_big

    return lshape, rshape, tuple(lslices), tuple(rslices)

def randitems_from_structure(fmt, t):
    """Return a list of random items for structure 't' with format
       'fmtchar'."""
    memlen, itemsize, _, _, _, _ = t
    return gen_items(memlen//itemsize, '#'+fmt, 'numpy')

def ndarray_from_structure(items, fmt, t, flags=0):
    """Return ndarray from the tuple returned by rand_structure()"""
    memlen, itemsize, ndim, shape, strides, offset = t
    return ndarray(items, shape=shape, strides=strides, format=fmt,
                   offset=offset, flags=ND_WRITABLE|flags)

def numpy_array_from_structure(items, fmt, t):
    """Return numpy_array from the tuple returned by rand_structure()"""
    memlen, itemsize, ndim, shape, strides, offset = t
    buf = bytearray(memlen)
    for j, v in enumerate(items):
        struct.pack_into(fmt, buf, j*itemsize, v)
    return numpy_array(buffer=buf, shape=shape, strides=strides,
                       dtype=fmt, offset=offset)


# ======================================================================
#                          memoryview casts
# ======================================================================

def cast_items(exporter, fmt, itemsize, shape=None):
    """Interpret the raw memory of 'exporter' as a list of items with
       size 'itemsize'. If shape=None, the new structure is assumed to
       be 1-D with n * itemsize = bytelen. If shape is given, the usual
       constraint for contiguous arrays prod(shape) * itemsize = bytelen
       applies. On success, return (items, shape). If the constraints
       cannot be met, return (None, None). If a chunk of bytes is interpreted
       as NaN as a result of float conversion, return ('nan', None)."""
    bytelen = exporter.nbytes
    if shape:
        if prod(shape) * itemsize != bytelen:
            return None, shape
    elif shape == []:
        if exporter.ndim == 0 or itemsize != bytelen:
            return None, shape
    else:
        n, r = divmod(bytelen, itemsize)
        shape = [n]
        if r != 0:
            return None, shape

    mem = exporter.tobytes()
    byteitems = [mem[i:i+itemsize] for i in range(0, len(mem), itemsize)]

    items = []
    for v in byteitems:
        item = struct.unpack(fmt, v)[0]
        if item != item:
            return 'nan', shape
        items.append(item)

    return (items, shape) if shape != [] else (items[0], shape)

def gencastshapes():
    """Generate shapes to test casting."""
    for n in range(32):
        yield [n]
    ndim = randrange(4, 6)
    minshape = 1 if randrange(100) > 80 else 2
    yield [randrange(minshape, 5) for _ in range(ndim)]
    ndim = randrange(2, 4)
    minshape = 1 if randrange(100) > 80 else 2
    yield [randrange(minshape, 5) for _ in range(ndim)]


# ======================================================================
#                              Actual tests
# ======================================================================

def genslices(n):
    """Generate all possible slices for a single dimension."""
    return product(range(-n, n+1), range(-n, n+1), range(-n, n+1))

def genslices_ndim(ndim, shape):
    """Generate all possible slice tuples for 'shape'."""
    iterables = [genslices(shape[n]) for n in range(ndim)]
    return product(*iterables)

def rslice(n, allow_empty=False):
    """Generate random slice for a single dimension of length n.
       If zero=True, the slices may be empty, otherwise they will
       be non-empty."""
    minlen = 0 if allow_empty or n == 0 else 1
    slicelen = randrange(minlen, n+1)
    return randslice_from_slicelen(slicelen, n)

def rslices(n, allow_empty=False):
    """Generate random slices for a single dimension."""
    for _ in range(5):
        yield rslice(n, allow_empty)

def rslices_ndim(ndim, shape, iterations=5):
    """Generate random slice tuples for 'shape'."""
    # non-empty slices
    for _ in range(iterations):
        yield tuple(rslice(shape[n]) for n in range(ndim))
    # possibly empty slices
    for _ in range(iterations):
        yield tuple(rslice(shape[n], allow_empty=True) for n in range(ndim))
    # invalid slices
    yield tuple(slice(0,1,0) for _ in range(ndim))

def rpermutation(iterable, r=None):
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    yield tuple(sample(pool, r))

def ndarray_print(nd):
    """Print ndarray for debugging."""
    try:
        x = nd.tolist()
    except (TypeError, NotImplementedError):
        x = nd.tobytes()
    if isinstance(nd, ndarray):
        offset = nd.offset
        flags = nd.flags
    else:
        offset = 'unknown'
        flags = 'unknown'
    print("ndarray(%s, shape=%s, strides=%s, suboffsets=%s, offset=%s, "
          "format='%s', itemsize=%s, flags=%s)" %
          (x, nd.shape, nd.strides, nd.suboffsets, offset,
           nd.format, nd.itemsize, flags))
    sys.stdout.flush()


ITERATIONS = 100
MAXDIM = 5
MAXSHAPE = 10

if SHORT_TEST:
    ITERATIONS = 10
    MAXDIM = 3
    MAXSHAPE = 4
    genslices = rslices
    genslices_ndim = rslices_ndim
    permutations = rpermutation


@unittest.skipUnless(struct, 'struct module required for this test.')
@unittest.skipUnless(ndarray, 'ndarray object required for this test')
class TestBufferProtocol(unittest.TestCase):

    def setUp(self):
        # The suboffsets tests need sizeof(void *).
        self.sizeof_void_p = get_sizeof_void_p()

    def verify(self, result, *, obj,
                     itemsize, fmt, readonly,
                     ndim, shape, strides,
                     lst, sliced=False, cast=False):
        # Verify buffer contents against expected values.
        if shape:
            expected_len = prod(shape)*itemsize
        else:
            if not fmt: # array has been implicitly cast to unsigned bytes
                expected_len = len(lst)
            else: # ndim = 0
                expected_len = itemsize

        # Reconstruct suboffsets from strides. Support for slicing
        # could be added, but is currently only needed for test_getbuf().
        suboffsets = ()
        if result.suboffsets:
            self.assertGreater(ndim, 0)

            suboffset0 = 0
            for n in range(1, ndim):
                if shape[n] == 0:
                    break
                if strides[n] <= 0:
                    suboffset0 += -strides[n] * (shape[n]-1)

            suboffsets = [suboffset0] + [-1 for v in range(ndim-1)]

            # Not correct if slicing has occurred in the first dimension.
            stride0 = self.sizeof_void_p
            if strides[0] < 0:
                stride0 = -stride0
            strides = [stride0] + list(strides[1:])

        self.assertIs(result.obj, obj)
        self.assertEqual(result.nbytes, expected_len)
        self.assertEqual(result.itemsize, itemsize)
        self.assertEqual(result.format, fmt)
        self.assertIs(result.readonly, readonly)
        self.assertEqual(result.ndim, ndim)
        self.assertEqual(result.shape, tuple(shape))
        if not (sliced and suboffsets):
            self.assertEqual(result.strides, tuple(strides))
        self.assertEqual(result.suboffsets, tuple(suboffsets))

        if isinstance(result, ndarray) or is_memoryview_format(fmt):
            rep = result.tolist() if fmt else result.tobytes()
            self.assertEqual(rep, lst)

        if not fmt: # array has been cast to unsigned bytes,
            return  # the remaining tests won't work.

        # PyBuffer_GetPointer() is the definition how to access an item.
        # If PyBuffer_GetPointer(indices) is correct for all possible
        # combinations of indices, the buffer is correct.
        #
        # Also test tobytes() against the flattened 'lst', with all items
        # packed to bytes.
        if not cast: # casts chop up 'lst' in different ways
            b = bytearray()
            buf_err = None
            for ind in indices(shape):
                try:
                    item1 = get_pointer(result, ind)
                    item2 = get_item(lst, ind)
                    if isinstance(item2, tuple):
                        x = struct.pack(fmt, *item2)
                    else:
                        x = struct.pack(fmt, item2)
                    b.extend(x)
                except BufferError:
                    buf_err = True # re-exporter does not provide full buffer
                    break
                self.assertEqual(item1, item2)

            if not buf_err:
                # test tobytes()
                self.assertEqual(result.tobytes(), b)

                # test hex()
                m = memoryview(result)
                h = "".join("%02x" % c for c in b)
                self.assertEqual(m.hex(), h)

                # lst := expected multi-dimensional logical representation
                # flatten(lst) := elements in C-order
                ff = fmt if fmt else 'B'
                flattened = flatten(lst)

                # Rules for 'A': if the array is already contiguous, return
                # the array unaltered. Otherwise, return a contiguous 'C'
                # representation.
                for order in ['C', 'F', 'A']:
                    expected = result
                    if order == 'F':
                        if not is_contiguous(result, 'A') or \
                           is_contiguous(result, 'C'):
                            # For constructing the ndarray, convert the
                            # flattened logical representation to Fortran order.
                            trans = transpose(flattened, shape)
                            expected = ndarray(trans, shape=shape, format=ff,
                                               flags=ND_FORTRAN)
                    else: # 'C', 'A'
                        if not is_contiguous(result, 'A') or \
                           is_contiguous(result, 'F') and order == 'C':
                            # The flattened list is already in C-order.
                            expected = ndarray(flattened, shape=shape, format=ff)

                    contig = get_contiguous(result, PyBUF_READ, order)
                    self.assertEqual(contig.tobytes(), b)
                    self.assertTrue(cmp_contig(contig, expected))

                    if ndim == 0:
                        continue

                    nmemb = len(flattened)
                    ro = 0 if readonly else ND_WRITABLE

                    ### See comment in test_py_buffer_to_contiguous for an
                    ### explanation why these tests are valid.

                    # To 'C'
                    contig = py_buffer_to_contiguous(result, 'C', PyBUF_FULL_RO)
                    self.assertEqual(len(contig), nmemb * itemsize)
                    initlst = [struct.unpack_from(fmt, contig, n*itemsize)
                               for n in range(nmemb)]
                    if len(initlst[0]) == 1:
                        initlst = [v[0] for v in initlst]

                    y = ndarray(initlst, shape=shape, flags=ro, format=fmt)
                    self.assertEqual(memoryview(y), memoryview(result))

                    contig_bytes = memoryview(result).tobytes()
                    self.assertEqual(contig_bytes, contig)

                    contig_bytes = memoryview(result).tobytes(order=None)
                    self.assertEqual(contig_bytes, contig)

                    contig_bytes = memoryview(result).tobytes(order='C')
                    self.assertEqual(contig_bytes, contig)

                    # To 'F'
                    contig = py_buffer_to_contiguous(result, 'F', PyBUF_FULL_RO)
                    self.assertEqual(len(contig), nmemb * itemsize)
                    initlst = [struct.unpack_from(fmt, contig, n*itemsize)
                               for n in range(nmemb)]
                    if len(initlst[0]) == 1:
                        initlst = [v[0] for v in initlst]

                    y = ndarray(initlst, shape=shape, flags=ro|ND_FORTRAN,
                                format=fmt)
                    self.assertEqual(memoryview(y), memoryview(result))

                    contig_bytes = memoryview(result).tobytes(order='F')
                    self.assertEqual(contig_bytes, contig)

                    # To 'A'
                    contig = py_buffer_to_contiguous(result, 'A', PyBUF_FULL_RO)
                    self.assertEqual(len(contig), nmemb * itemsize)
                    initlst = [struct.unpack_from(fmt, contig, n*itemsize)
                               for n in range(nmemb)]
                    if len(initlst[0]) == 1:
                        initlst = [v[0] for v in initlst]

                    f = ND_FORTRAN if is_contiguous(result, 'F') else 0
                    y = ndarray(initlst, shape=shape, flags=f|ro, format=fmt)
                    self.assertEqual(memoryview(y), memoryview(result))

                    contig_bytes = memoryview(result).tobytes(order='A')
                    self.assertEqual(contig_bytes, contig)

        if                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                w)

        # zeros in shape, struct module
        nd1 = ndarray([900, 961], shape=[0], format='= h0c')
        nd2 = ndarray([-900, -961], shape=[0], format='@   i')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

    def test_memoryview_compare_zero_strides(self):

        # zero strides
        nd1 = ndarray([900, 900, 900, 900], shape=[4], format='@L')
        nd2 = ndarray([900], shape=[4], strides=[0], format='L')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

        # zero strides, struct module
        nd1 = ndarray([(900, 900)]*4, shape=[4], format='@ Li')
        nd2 = ndarray([(900, 900)], shape=[4], strides=[0], format='!L  h')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

    def test_memoryview_compare_random_formats(self):

        # random single character native formats
        n = 10
        for char in fmtdict['@m']:
            fmt, items, singleitem = randitems(n, 'memoryview', '@', char)
            for flags in (0, ND_PIL):
                nd = ndarray(items, shape=[n], format=fmt, flags=flags)
                m = memoryview(nd)
                self.assertEqual(m, nd)

                nd = nd[::-3]
                m = memoryview(nd)
                self.assertEqual(m, nd)

        # random formats
        n = 10
        for _ in range(100):
            fmt, items, singleitem = randitems(n)
            for flags in (0, ND_PIL):
                nd = ndarray(items, shape=[n], format=fmt, flags=flags)
                m = memoryview(nd)
                self.assertEqual(m, nd)

                nd = nd[::-3]
                m = memoryview(nd)
                self.assertEqual(m, nd)

    def test_memoryview_compare_multidim_c(self):

        # C-contiguous, different values
        nd1 = ndarray(list(range(-15, 15)), shape=[3, 2, 5], format='@h')
        nd2 = ndarray(list(range(0, 30)), shape=[3, 2, 5], format='@h')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # C-contiguous, different values, struct module
        nd1 = ndarray([(0, 1, 2)]*30, shape=[3, 2, 5], format='=f q xxL')
        nd2 = ndarray([(-1.2, 1, 2)]*30, shape=[3, 2, 5], format='< f 2Q')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # C-contiguous, different shape
        nd1 = ndarray(list(range(30)), shape=[2, 3, 5], format='L')
        nd2 = ndarray(list(range(30)), shape=[3, 2, 5], format='L')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # C-contiguous, different shape, struct module
        nd1 = ndarray([(0, 1, 2)]*21, shape=[3, 7], format='! b B xL')
        nd2 = ndarray([(0, 1, 2)]*21, shape=[7, 3], format='= Qx l xxL')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # C-contiguous, different format, struct module
        nd1 = ndarray(list(range(30)), shape=[2, 3, 5], format='L')
        nd2 = ndarray(list(range(30)), shape=[2, 3, 5], format='l')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

    def test_memoryview_compare_multidim_fortran(self):

        # Fortran-contiguous, different values
        nd1 = ndarray(list(range(-15, 15)), shape=[5, 2, 3], format='@h',
                      flags=ND_FORTRAN)
        nd2 = ndarray(list(range(0, 30)), shape=[5, 2, 3], format='@h',
                      flags=ND_FORTRAN)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # Fortran-contiguous, different values, struct module
        nd1 = ndarray([(2**64-1, -1)]*6, shape=[2, 3], format='=Qq',
                      flags=ND_FORTRAN)
        nd2 = ndarray([(-1, 2**64-1)]*6, shape=[2, 3], format='=qQ',
                      flags=ND_FORTRAN)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # Fortran-contiguous, different shape
        nd1 = ndarray(list(range(-15, 15)), shape=[2, 3, 5], format='l',
                      flags=ND_FORTRAN)
        nd2 = ndarray(list(range(-15, 15)), shape=[3, 2, 5], format='l',
                      flags=ND_FORTRAN)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # Fortran-contiguous, different shape, struct module
        nd1 = ndarray(list(range(-15, 15)), shape=[2, 3, 5], format='0ll',
                      flags=ND_FORTRAN)
        nd2 = ndarray(list(range(-15, 15)), shape=[3, 2, 5], format='l',
                      flags=ND_FORTRAN)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # Fortran-contiguous, different format, struct module
        nd1 = ndarray(list(range(30)), shape=[5, 2, 3], format='@h',
                      flags=ND_FORTRAN)
        nd2 = ndarray(list(range(30)), shape=[5, 2, 3], format='@b',
                      flags=ND_FORTRAN)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

    def test_memoryview_compare_multidim_mixed(self):

        # mixed C/Fortran contiguous
        lst1 = list(range(-15, 15))
        lst2 = transpose(lst1, [3, 2, 5])
        nd1 = ndarray(lst1, shape=[3, 2, 5], format='@l')
        nd2 = ndarray(lst2, shape=[3, 2, 5], format='l', flags=ND_FORTRAN)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, w)

        # mixed C/Fortran contiguous, struct module
        lst1 = [(-3.3, -22, b'x')]*30
        lst1[5] = (-2.2, -22, b'x')
        lst2 = transpose(lst1, [3, 2, 5])
        nd1 = ndarray(lst1, shape=[3, 2, 5], format='d b c')
        nd2 = ndarray(lst2, shape=[3, 2, 5], format='d h c', flags=ND_FORTRAN)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, w)

        # different values, non-contiguous
        ex1 = ndarray(list(range(40)), shape=[5, 8], format='@I')
        nd1 = ex1[3:1:-1, ::-2]
        ex2 = ndarray(list(range(40)), shape=[5, 8], format='I')
        nd2 = ex2[1:3:1, ::-2]
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # same values, non-contiguous, struct module
        ex1 = ndarray([(2**31-1, -2**31)]*22, shape=[11, 2], format='=ii')
        nd1 = ex1[3:1:-1, ::-2]
        ex2 = ndarray([(2**31-1, -2**31)]*22, shape=[11, 2], format='>ii')
        nd2 = ex2[1:3:1, ::-2]
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

        # different shape
        ex1 = ndarray(list(range(30)), shape=[2, 3, 5], format='b')
        nd1 = ex1[1:3:, ::-2]
        nd2 = ndarray(list(range(30)), shape=[3, 2, 5], format='b')
        nd2 = ex2[1:3:, ::-2]
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # different shape, struct module
        ex1 = ndarray(list(range(30)), shape=[2, 3, 5], format='B')
        nd1 = ex1[1:3:, ::-2]
        nd2 = ndarray(list(range(30)), shape=[3, 2, 5], format='b')
        nd2 = ex2[1:3:, ::-2]
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # different format, struct module
        ex1 = ndarray([(2, b'123')]*30, shape=[5, 3, 2], format='b3s')
        nd1 = ex1[1:3:, ::-2]
        nd2 = ndarray([(2, b'123')]*30, shape=[5, 3, 2], format='i3s')
        nd2 = ex2[1:3:, ::-2]
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

    def test_memoryview_compare_multidim_zero_shape(self):

        # zeros in shape
        nd1 = ndarray(list(range(30)), shape=[0, 3, 2], format='i')
        nd2 = ndarray(list(range(30)), shape=[5, 0, 2], format='@i')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # zeros in shape, struct module
        nd1 = ndarray(list(range(30)), shape=[0, 3, 2], format='i')
        nd2 = ndarray(list(range(30)), shape=[5, 0, 2], format='@i')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

    def test_memoryview_compare_multidim_zero_strides(self):

        # zero strides
        nd1 = ndarray([900]*80, shape=[4, 5, 4], format='@L')
        nd2 = ndarray([900], shape=[4, 5, 4], strides=[0, 0, 0], format='L')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)
        self.assertEqual(v.tolist(), w.tolist())

        # zero strides, struct module
        nd1 = ndarray([(1, 2)]*10, shape=[2, 5], format='=lQ')
        nd2 = ndarray([(1, 2)], shape=[2, 5], strides=[0, 0], format='<lQ')
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

    def test_memoryview_compare_multidim_suboffsets(self):

        # suboffsets
        ex1 = ndarray(list(range(40)), shape=[5, 8], format='@I')
        nd1 = ex1[3:1:-1, ::-2]
        ex2 = ndarray(list(range(40)), shape=[5, 8], format='I', flags=ND_PIL)
        nd2 = ex2[1:3:1, ::-2]
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # suboffsets, struct module
        ex1 = ndarray([(2**64-1, -1)]*40, shape=[5, 8], format='=Qq',
                      flags=ND_WRITABLE)
        ex1[2][7] = (1, -2)
        nd1 = ex1[3:1:-1, ::-2]

        ex2 = ndarray([(2**64-1, -1)]*40, shape=[5, 8], format='>Qq',
                      flags=ND_PIL|ND_WRITABLE)
        ex2[2][7] = (1, -2)
        nd2 = ex2[1:3:1, ::-2]

        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

        # suboffsets, different shape
        ex1 = ndarray(list(range(30)), shape=[2, 3, 5], format='b',
                      flags=ND_PIL)
        nd1 = ex1[1:3:, ::-2]
        nd2 = ndarray(list(range(30)), shape=[3, 2, 5], format='b')
        nd2 = ex2[1:3:, ::-2]
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # suboffsets, different shape, struct module
        ex1 = ndarray([(2**8-1, -1)]*40, shape=[2, 3, 5], format='Bb',
                      flags=ND_PIL|ND_WRITABLE)
        nd1 = ex1[1:2:, ::-2]

        ex2 = ndarray([(2**8-1, -1)]*40, shape=[3, 2, 5], format='Bb')
        nd2 = ex2[1:2:, ::-2]

        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # suboffsets, different format
        ex1 = ndarray(list(range(30)), shape=[5, 3, 2], format='i', flags=ND_PIL)
        nd1 = ex1[1:3:, ::-2]
        ex2 = ndarray(list(range(30)), shape=[5, 3, 2], format='@I', flags=ND_PIL)
        nd2 = ex2[1:3:, ::-2]
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, nd2)
        self.assertEqual(w, nd1)
        self.assertEqual(v, w)

        # suboffsets, different format, struct module
        ex1 = ndarray([(b'hello', b'', 1)]*27, shape=[3, 3, 3], format='5s0sP',
                      flags=ND_PIL|ND_WRITABLE)
        ex1[1][2][2] = (b'sushi', b'', 1)
        nd1 = ex1[1:3:, ::-2]

        ex2 = ndarray([(b'hello', b'', 1)]*27, shape=[3, 3, 3], format='5s0sP',
                      flags=ND_PIL|ND_WRITABLE)
        ex1[1][2][2] = (b'sushi', b'', 1)
        nd2 = ex2[1:3:, ::-2]

        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertNotEqual(v, nd2)
        self.assertNotEqual(w, nd1)
        self.assertNotEqual(v, w)

        # initialize mixed C/Fortran + suboffsets
        lst1 = list(range(-15, 15))
        lst2 = transpose(lst1, [3, 2, 5])
        nd1 = ndarray(lst1, shape=[3, 2, 5], format='@l', flags=ND_PIL)
        nd2 = ndarray(lst2, shape=[3, 2, 5], format='l', flags=ND_FORTRAN|ND_PIL)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, w)

        # initialize mixed C/Fortran + suboffsets, struct module
        lst1 = [(b'sashimi', b'sliced', 20.05)]*30
        lst1[11] = (b'ramen', b'spicy', 9.45)
        lst2 = transpose(lst1, [3, 2, 5])

        nd1 = ndarray(lst1, shape=[3, 2, 5], format='< 10p 9p d', flags=ND_PIL)
        nd2 = ndarray(lst2, shape=[3, 2, 5], format='> 10p 9p d',
                      flags=ND_FORTRAN|ND_PIL)
        v = memoryview(nd1)
        w = memoryview(nd2)

        self.assertEqual(v, nd1)
        self.assertEqual(w, nd2)
        self.assertEqual(v, w)

    def test_memoryview_compare_not_equal(self):

        # items not equal
        for byteorder in ['=', '<', '>', '!']:
            x = ndarray([2**63]*120, shape=[3,5,2,2,2], format=byteorder+'Q')
            y = ndarray([2**63]*120, shape=[3,5,2,2,2], format=byteorder+'Q',
                        flags=ND_WRITABLE|ND_FORTRAN)
            y[2][3][1][1][1] = 1
            a = memoryview(x)
            b = memoryview(y)
            self.assertEqual(a, x)
            self.assertEqual(b, y)
            self.assertNotEqual(a, b)
            self.assertNotEqual(a, y)
            self.assertNotEqual(b, x)

            x = ndarray([(2**63, 2**31, 2**15)]*120, shape=[3,5,2,2,2],
                        format=byteorder+'QLH')
            y = ndarray([(2**63, 2**31, 2**15)]*120, shape=[3,5,2,2,2],
                        format=byteorder+'QLH', flags=ND_WRITABLE|ND_FORTRAN)
            y[2][3][1][1][1] = (1, 1, 1)
            a = memoryview(x)
            b = memoryview(y)
            self.assertEqual(a, x)
            self.assertEqual(b, y)
            self.assertNotEqual(a, b)
            self.assertNotEqual(a, y)
            self.assertNotEqual(b, x)

    def test_memoryview_check_released(self):

        a = array.array('d', [1.1, 2.2, 3.3])

        m = memoryview(a)
        m.release()

        # PyMemoryView_FromObject()
        self.assertRaises(ValueError, memoryview, m)
        # memoryview.cast()
        self.assertRaises(ValueError, m.cast, 'c')
        # getbuffer()
        self.assertRaises(ValueError, ndarray, m)
        # memoryview.tolist()
        self.assertRaises(ValueError, m.tolist)
        # memoryview.tobytes()
        self.assertRaises(ValueError, m.tobytes)
        # sequence
        self.assertRaises(ValueError, eval, "1.0 in m", locals())
        # subscript
        self.assertRaises(ValueError, m.__getitem__, 0)
        # assignment
        self.assertRaises(ValueError, m.__setitem__, 0, 1)

        for attr in ('obj', 'nbytes', 'readonly', 'itemsize', 'format', 'ndim',
                     'shape', 'strides', 'suboffsets', 'c_contiguous',
                     'f_contiguous', 'contiguous'):
            self.assertRaises(ValueError, m.__getattribute__, attr)

        # richcompare
        b = array.array('d', [1.1, 2.2, 3.3])
        m1 = memoryview(a)
        m2 = memoryview(b)

        self.assertEqual(m1, m2)
        m1.release()
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, a)
        self.assertEqual(m1, m1)

    def test_memoryview_tobytes(self):
        # Many implicit tests are already in self.verify().

        t = (-529, 576, -625, 676, -729)

        nd = ndarray(t, shape=[5], format='@h')
        m = memoryview(nd)
        self.assertEqual(m, nd)
        self.assertEqual(m.tobytes(), nd.tobytes())

        nd = ndarray([t], shape=[1], format='>hQiLl')
        m = memoryview(nd)
        self.assertEqual(m, nd)
        self.assertEqual(m.tobytes(), nd.tobytes())

        nd = ndarray([t for _ in range(12)], shape=[2,2,3], format='=hQiLl')
        m = memoryview(nd)
        self.assertEqual(m, nd)
        self.assertEqual(m.tobytes(), nd.tobytes())

        nd = ndarray([t for _ in range(120)], shape=[5,2,2,3,2],
                     format='<hQiLl')
        m = memoryview(nd)
        self.assertEqual(m, nd)
        self.assertEqual(m.tobytes(), nd.tobytes())

        # Unknown formats are handled: tobytes() purely depends on itemsize.
        if ctypes:
            # format: "T{>l:x:>l:y:}"
            class BEPoint(ctypes.BigEndianStructure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            point = BEPoint(100, 200)
            a = memoryview(point)
            self.assertEqual(a.tobytes(), bytes(point))

    def test_memoryview_get_contiguous(self):
        # Many implicit tests are already in self.verify().

        # no buffer interface
        self.assertRaises(TypeError, get_contiguous, {}, PyBUF_READ, 'F')

        # writable request to read-only object
        self.assertRaises(BufferError, get_contiguous, b'x', PyBUF_WRITE, 'C')

        # writable request to non-contiguous object
        nd = ndarray([1, 2, 3], shape=[2], strides=[2])
        self.assertRaises(BufferError, get_contiguous, nd, PyBUF_WRITE, 'A')

        # scalar, read-only request from read-only exporter
        nd = ndarray(9, shape=(), format="L")
        for order in ['C', 'F', 'A']:
            m = get_contiguous(nd, PyBUF_READ, order)
            self.assertEqual(m, nd)
            self.assertEqual(m[()], 9)

        # scalar, read-only request from writable exporter
        nd = ndarray(9, shape=(), format="L", flags=ND_WRITABLE)
        for order in ['C', 'F', 'A']:
            m = get_contiguous(nd, PyBUF_READ, order)
            self.assertEqual(m, nd)
            self.assertEqual(m[()], 9)

        # scalar, writable request
        for order in ['C', 'F', 'A']:
            nd[()] = 9
            m = get_contiguous(nd, PyBUF_WRITE, order)
            self.assertEqual(m, nd)
            self.assertEqual(m[()], 9)

            m[()] = 10
            self.assertEqual(m[()], 10)
            self.assertEqual(nd[()], 10)

        # zeros in shape
        nd = ndarray([1], shape=[0], format="L", flags=ND_WRITABLE)
        for order in ['C', 'F', 'A']:
            m = get_contiguous(nd, PyBUF_READ, order)
            self.assertRaises(IndexError, m.__getitem__, 0)
            self.assertEqual(m, nd)
            self.assertEqual(m.tolist(), [])

        nd = ndarray(list(range(8)), shape=[2, 0, 7], format="L",
                     flags=ND_WRITABLE)
        for order in ['C', 'F', 'A']:
            m = get_contiguous(nd, PyBUF_READ, order)
            self.assertEqual(ndarray(m).tolist(), [[], []])

        # one-dimensional
        nd = ndarray([1], shape=[1], format="h", flags=ND_WRITABLE)
        for order in ['C', 'F', 'A']:
            m = get_contiguous(nd, PyBUF_WRITE, order)
            self.assertEqual(m, nd)
            self.assertEqual(m.tolist(), nd.tolist())

        nd = ndarray([1, 2, 3], shape=[3], format="b", flags=ND_WRITABLE)
        for order in ['C', 'F', 'A']:
            m = get_contiguous(nd, PyBUF_WRITE, order)
            self.assertEqual(m, nd)
            self.assertEqual(m.tolist(), nd.tolist())

        # one-dimensional, non-contiguous
        nd = ndarray([1, 2, 3], shape=[2], strides=[2], flags=ND_WRITABLE)
        for order in ['C', 'F', 'A']:
            m = get_contiguous(nd, PyBUF_READ, order)
            self.assertEqual(m, nd)
            self.assertEqual(m.tolist(), nd.tolist())
            self.assertRaises(TypeError, m.__setitem__, 1, 20)
            self.assertEqual(m[1], 3)
            self.assertEqual(nd[1], 3)

        nd = nd[::-1]
        for order in ['C', 'F', 'A']:
            m = get_contiguous(nd, PyBUF_READ, order)
            self.assertEqual(m, nd)
            self.assertEqual(m.tolist(), nd.tolist())
            self.assertRaises(TypeError, m.__setitem__, 1, 20)
            self.assertEqual(m[1], 1)
            self.assertEqual(nd[1], 1)

        # multi-dimensional, contiguous input
        nd = ndarray(list(range(12)), shape=[3, 4], flags=ND_WRITABLE)
        for order in ['C', 'A']:
            m = get_contiguous(nd, PyBUF_WRITE, order)
            self.assertEqual(ndarray(m).tolist(), nd.tolist())

        self.assertRaises(BufferError, get_contiguous, nd, PyBUF_WRITE, 'F')
        m = get_contiguous(nd, PyBUF_READ, order)
        self.assertEqual(ndarray(m).tolist(), nd.tolist())

        nd = ndarray(list(range(12)), shape=[3, 4],
                     flags=ND_WRITABLE|ND_FORTRAN)
        for order in ['F', 'A']:
            m = get_contiguous(nd, PyBUF_WRITE, order)
            self.assertEqual(ndarray(m).tolist(), nd.tolist())

        self.assertRaises(BufferError, get_contiguous, nd, PyBUF_WRITE, 'C')
        m = get_contiguous(nd, PyBUF_READ, order)
        self.assertEqual(ndarray(m).tolist(), nd.tolist())

        # multi-dimensional, non-contiguous input
        nd = ndarray(list(range(12)), shape=[3, 4], flags=ND_WRITABLE|ND_PIL)
        for order in ['C', 'F', 'A']:
            self.assertRaises(BufferError, get_contiguous, nd, PyBUF_WRITE,
                              order)
            m = get_contiguous(nd, PyBUF_READ, order)
            self.assertEqual(ndarray(m).tolist(), nd.tolist())

        # flags
        nd = ndarray([1,2,3,4,5], shape=[3], strides=[2])
        m = get_contiguous(nd, PyBUF_READ, 'C')
        self.assertTrue(m.c_contiguous)

    def test_memoryview_serializing(self):

        # C-contiguous
        size = struct.calcsize('i')
        a = array.array('i', [1,2,3,4,5])
        m = memoryview(a)
        buf = io.BytesIO(m)
        b = bytearray(5*size)
        buf.readinto(b)
        self.assertEqual(m.tobytes(), b)

        # C-contiguous, multi-dimensional
        size = struct.calcsize('L')
        nd = ndarray(list(range(12)), shape=[2,3,2], format="L")
        m = memoryview(nd)
        buf = io.BytesIO(m)
        b = bytearray(2*3*2*size)
        buf.readinto(b)
        self.assertEqual(m.tobytes(), b)

        # Fortran contiguous, multi-dimensional
        #size = struct.calcsize('L')
        #nd = ndarray(list(range(12)), shape=[2,3,2], format="L",
        #             flags=ND_FORTRAN)
        #m = memoryview(nd)
        #buf = io.BytesIO(m)
        #b = bytearray(2*3*2*size)
        #buf.readinto(b)
        #self.assertEqual(m.tobytes(), b)

    def test_memoryview_hash(self):

        # bytes exporter
        b = bytes(list(range(12)))
        m = memoryview(b)
        self.assertEqual(hash(b), hash(m))

        # C-contiguous
        mc = m.cast('c', shape=[3,4])
        self.assertEqual(hash(mc), hash(b))

        # non-contiguous
        mx = m[::-2]
        b = bytes(list(range(12))[::-2])
        self.assertEqual(hash(mx), hash(b))

        # Fortran contiguous
        nd = ndarray(list(range(30)), shape=[3,2,5], flags=ND_FORTRAN)
        m = memoryview(nd)
        self.assertEqual(hash(m), hash(nd))

        # multi-dimensional slice
        nd = ndarray(list(range(30)), shape=[3,2,5])
        x = nd[::2, ::, ::-1]
        m = memoryview(x)
        self.assertEqual(hash(m), hash(x))

        # multi-dimensional slice with suboffsets
        nd = ndarray(list(range(30)), shape=[2,5,3], flags=ND_PIL)
        x = nd[::2, ::, ::-1]
        m = memoryview(x)
        self.assertEqual(hash(m), hash(x))

        # equality-hash invariant
        x = ndarray(list(range(12)), shape=[12], format='B')
        a = memoryview(x)

        y = ndarray(list(range(12)), shape=[12], format='b')
        b = memoryview(y)

        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

        # non-byte formats
        nd = ndarray(list(range(12)), shape=[2,2,3], format='L')
        m = memoryview(nd)
        self.assertRaises(ValueError, m.__hash__)

        nd = ndarray(list(range(-6, 6)), shape=[2,2,3], format='h')
        m = memoryview(nd)
        self.assertRaises(ValueError, m.__hash__)

        nd = ndarray(list(range(12)), shape=[2,2,3], format='= L')
        m = memoryview(nd)
        self.assertRaises(ValueError, m.__hash__)

        nd = ndarray(list(range(-6, 6)), shape=[2,2,3], format='< h')
        m = memoryview(nd)
        self.assertRaises(ValueError, m.__hash__)

    def test_memoryview_release(self):

        # Create re-exporter from getbuffer(memoryview), then release the view.
        a = bytearray([1,2,3])
        m = memoryview(a)
        nd = ndarray(m) # re-exporter
        self.assertRaises(BufferError, m.release)
        del nd
        m.release()

        a = bytearray([1,2,3])
        m = memoryview(a)
        nd1 = ndarray(m, getbuf=PyBUF_FULL_RO, flags=ND_REDIRECT)
        nd2 = ndarray(nd1, getbuf=PyBUF_FULL_RO, flags=ND_REDIRECT)
        self.assertIs(nd2.obj, m)
        self.assertRaises(BufferError, m.release)
        del nd1, nd2
        m.release()

        # chained views
        a = bytearray([1,2,3])
        m1 = memoryview(a)
        m2 = memoryview(m1)
        nd = ndarray(m2) # re-exporter
        m1.release()
        self.assertRaises(BufferError, m2.release)
        del nd
        m2.release()

        a = bytearray([1,2,3])
        m1 = memoryview(a)
        m2 = memoryview(m1)
        nd1 = ndarray(m2, getbuf=PyBUF_FULL_RO, flags=ND_REDIRECT)
        nd2 = ndarray(nd1, getbuf=PyBUF_FULL_RO, flags=ND_REDIRECT)
        self.assertIs(nd2.obj, m2)
        m1.release()
        self.assertRaises(BufferError, m2.release)
        del nd1, nd2
        m2.release()

        # Allow changing layout while buffers are exported.
        nd = ndarray([1,2,3], shape=[3], flags=ND_VAREXPORT)
        m1 = memoryview(nd)

        nd.push([4,5,6,7,8], shape=[5]) # mutate nd
        m2 = memoryview(nd)

        x = memoryview(m1)
        self.assertEqual(x.tolist(), m1.tolist())

        y = memoryview(m2)
        self.assertEqual(y.tolist(), m2.tolist())
        self.assertEqual(y.tolist(), nd.tolist())
        m2.release()
        y.release()

        nd.pop() # pop the current view
        self.assertEqual(x.tolist(), nd.tolist())

        del nd
        m1.release()
        x.release()

        # If multiple memoryviews share the same managed buffer, implicit
        # release() in the context manager's __exit__() method should still
        # work.
        def catch22(b):
            with memoryview(b) as m2:
                pass

        x = bytearray(b'123')
        with memoryview(x) as m1:
            catch22(m1)
            self.assertEqual(m1[0], ord(b'1'))

        x = ndarray(list(range(12)), shape=[2,2,3], format='l')
        y = ndarray(x, getbuf=PyBUF_FULL_RO, flags=ND_REDIRECT)
        z = ndarray(y, getbuf=PyBUF_FULL_RO, flags=ND_REDIRECT)
        self.assertIs(z.obj, x)
        with memoryview(z) as m:
            catch22(m)
            self.assertEqual(m[0:1].tolist(), [[[0, 1, 2], [3, 4, 5]]])

        # Test garbage collection.
        for flags in (0, ND_REDIRECT):
            x = bytearray(b'123')
            with memoryview(x) as m1:
                del x
                y = ndarray(m1, getbuf=PyBUF_FULL_RO, flags=flags)
                with memoryview(y) as m2:
                    del y
                    z = ndarray(m2, getbuf=PyBUF_FULL_RO, flags=flags)
                    with memoryview(z) as m3:
                        del z
                        catch22(m3)
                        catch22(m2)
                        catch22(m1)
                        self.assertEqual(m1[0], ord(b'1'))
                        self.assertEqual(m2[1], ord(b'2'))
                        self.assertEqual(m3[2], ord(b'3'))
                        del m3
                    del m2
                del m1

            x = bytearray(b'123')
            with memoryview(x) as m1:
                del x
                y = ndarray(m1, getbuf=PyBUF_FULL_RO, flags=flags)
                with memoryview(y) as m2:
                    del y
                    z = ndarray(m2, getbuf=PyBUF_FULL_RO, flags=flags)
                    with memoryview(z) as m3:
                        del z
                        catch22(m1)
                        catch22(m2)
                        catch22(m3)
                        self.assertEqual(m1[0], ord(b'1'))
                        self.assertEqual(m2[1], ord(b'2'))
                        self.assertEqual(m3[2], ord(b'3'))
                        del m1, m2, m3

        # memoryview.release() fails if the view has exported buffers.
        x = bytearray(b'123')
        with self.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        