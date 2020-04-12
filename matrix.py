#!/usr/bin/env python

from operator import mul, add
from types import IntType
from fractions import Fraction as QQ

class DoesNotExistError(Exception):
    '''We raise this exception whenever we're told to perform a
    mathematically malformed operation.'''
    def __init__(self, msg): self.msg = msg

    def __str__(self):
        return repr(self.msg)

class Matrix(object):
    '''Representation of a matrix with entries from an arbitrary
    field. Might extend to support arbitrary commutative rings.'''
    def __init__(self, field, row_vectors): 
        if not self._is_well_formed(row_vectors):
            raise DoesNotExistError('Malformed matrix.')

        self._field = field
        self._rows = [map(self._field, i) for i in row_vectors]
        self._columns = None
        self._det = None
        self._rref = None

    def rows(self):
        return self._rows

    def columns(self):
        # lazily compute the columns
        if not hasattr(self, '_columns'):
            rows = self.rows()
            self._columns = [[i[j] for i in rows]
                             for j in xrange(len(rows[0]))]
        return self._columns

    def field(self):
        return self._field

    def is_square(self):
        '''Return True if this matrix is square.'''
        return len(self.rows()) == len(self.columns())

    def transpose(self):
        '''Return the transpose of this matrix.'''
        return Matrix(self.field(), self.columns()) 

    def swap_rows(self, i, j):
        '''Swap row i with row j and yield the resulting
        matrix. Zero-indexed.'''
        rows = list(self.rows()) # don't use the reference
        rows[i], rows[j] = rows[j], rows[i]
        return Matrix(self.field(), rows)

    def swap_columns(self, i, j):
        return self.transpose().swap_rows(i, j).transpose()

    def scale_row(self, i, a):
        '''Scale row i by a. Zero-indexed.'''
        rows = list(self.rows())
        scaled = [self.field()(a) * j for j in rows[i]]
        rows[i] = scaled
        return Matrix(self.field(), rows)

    def scale_and_add_row(self, i, a, j):
        '''Scale row i by a and add it to row j. Zero-indexed.'''
        rows = list(self.rows())
        scaled = [self.field()(a) * k for k in rows[i]]
        rows[j] = map(add, scaled, rows[j])
        return Matrix(self.field(), rows)

    # alias det to determinant
    det = lambda s: s.determinant()

    def determinant(self):
        '''Return the determinant of this matrix. Raises DoesNotExist
        if matrix is not square.'''
        if not self.is_square():
            raise DoesNotExistError('Attempt to take determinant of non-square matrix.')

        if not hasattr(self, '_det'):
            self._rref, self._det = self._gauss_jordan()

        return self._det

    def _is_well_formed(self, matrix):
        '''Ensure the row vectors are of uniform length.'''
        r = map(len, matrix)
        return (not len(r)) or (r.count(r[0]) == len(r))

    def __mul__(self, matrix):
        '''Matrix multiplication from the left.'''
        if not isinstance(matrix.field(), type(self.field())):
            raise DoesNotExistError('Mismatched fields in multiplication.')

        if len(self.columns()) != len(matrix.rows()):
            raise DoesNotExistError('Mismatched dimensions in multiplication.')

        return Matrix(self.field(), [[sum(map(mul, row, column)) 
                                      for row in self.rows()] 
                                     for column in matrix.columns()])

    def rref(self):
        if not hasattr(self, '_rref'):
            self._rref, self._det = self._gauss_jordan()
        return self._rref

    def _find_pivot(self, x, from_index=0):
        '''Return the first nonzero entry in the list, with its position.'''
        if from_index >= len(x):
            raise IndexError('from_index larger than list')
        
        for i in xrange(from_index, len(x)):
            if x[i] != self.field()(0):
                return (x[i], i)

        return (None, None)

    def _gauss_jordan(self):
        '''Perform the Gauss-Jordan algorithm to yield the reduced-row
        echelon form of the current matrix coupled with the
        determinant if it has one.''' 

        matrix = self
        det = self.field()(1)
        row_index = column_index = 0

        rows_len = len(matrix.rows())
        columns_len = len(matrix.columns())
        
        while column_index < columns_len and row_index < rows_len: 
            # find a pivot
            pivot, position = self._find_pivot(matrix.columns()[column_index], 
                                               from_index=row_index)

            # no pivots
            if None == pivot:
                column_index += 1
                continue

            # bring pivot into position
            if position != row_index:
                matrix = matrix.swap_rows(position, row_index)
                det *= self.field()(-1)

            # normalize the pivot
            if pivot != self.field()(1):
                det *= pivot
                matrix = matrix.scale_row(row_index, self.field()(1)/pivot)
            
            # eliminate this column
            for i in xrange(rows_len):
                if i == row_index: # skip pivot row
                    continue
                if matrix.columns()[column_index][i] == self.field()(0): # skip zeros
                    continue
                matrix = matrix.scale_and_add_row(row_index, 
                                                  self.field()(-1)*matrix.columns()[column_index][i], i)
            column_index += 1
            row_index += 1

        # finish calculating the determinant
        if matrix.is_square():
            for i in xrange(rows_len):
                if matrix.rows()[i][i] == self.field()(0):
                    det = self.field()(0)
                    break
        else:
            det = None

        return matrix, det

    def __str__(self):
        '''This is hideous. Someone who isn't me should fix it.'''
        largest = str(max(map(len, map(str, reduce(add, self.rows())))))
        return '\n'.join(['[' + ' '.join([(('%'+largest+'s') % str(i)) 
                                          for i in row]) + ']' 
                          for row in self.rows()])

def _xgcd(a, b):
    if 0 == b:
        return 1, 0
    else:
        q, r = a / b, a % b
        s, t = _xgcd(b, r)
        return t, s - q * t

def _normalize_residue(r, m):
    return r % m if r >= m else _normalize_residue(r+m, m)

def _modular_inverse(x, m):
    return _normalize_residue(_xgcd(x, m)[0], m)

def GF(p):
    '''Factory method to generate Z/pZ for prime p.'''
    class PrimeField(object):
        '''Template class to return from this factory method.'''
        def __init__(self, value=0):
            if type(value) == IntType:
                self._value = value % self.char()
            elif hasattr(value, '_value'):
                self._value = value._value
            else:
                raise TypeError('Only ints or PrimeFields')

        char = lambda s: s.characteristic()

        def characteristic(self):
            return self._char
            
        def __eq__(self, x):
            if hasattr(x, '_value'):
                return self._value == x._value
            else:
                return NotImplemented
            
        def __ne__(self, x):
            r = self.__eq__(x)
            if r is not NotImplemented:
                return not r
            return NotImplemented
        
        def __mul__(self, x):
            return GF(self.char())(self._value * GF(self.char())(x)._value)
        
        def __add__(self, x):
            return GF(self.char())(self._value + GF(self.char())(x)._value)
        
        def __div__(self, x):
            if type(x) == IntType:
                if x == 0:
                    raise ZeroDivisionError
            elif hasattr(x, '_value'):
                if x._value == 0:
                    raise ZeroDivisionError
                
            return self * GF(self.char())(self._inverses[GF(self.char())(x)._value])

        def __sub__(self, x):
            return self + (GF(self.char())(-1) * GF(self.char())(x)) 
        
        def __str__(self):
            return str(self._value)
 
    setattr(PrimeField, '_char', p)
    setattr(PrimeField, '_inverses', map(lambda x: _modular_inverse(x, p), xrange(p)))
    return PrimeField

if __name__ == '__main__':
    m = Matrix(QQ, [[1,2,3], [4,5,6], [7,8,9]])
    print m
    print 'RREF:'
    print m.rref()
    print 'Determinant:', m.det()

    print
    p = Matrix(QQ, [[-2,2,3], [-1,1,3], [2,0,-1]])
    print p
    print 'RREF:'
    print p.rref()
    print 'Determinant:', p.det()

    print
    n = Matrix(GF(7), [[4,1,2,0], [1,6,3,3], [4,0,5,4]])
    print n
    print 'RREF:'
    print n.rref()

    print
    o = Matrix(GF(7), [[0,4,3], [2,4,0], [3,0,6], [0,2,3]])
    print o
    print 'RREF:'
    print o.rref()

    print
    q = Matrix(GF(7), [[0,0,5,0,0], [3,0,0,2,5], [5,1,1,6,1], [5,2,0,6,4]])
    print q
    print 'RREF:'
    print q.rref()

    print
    r = Matrix(GF(5), [[0,0,5,0], [3,0,0,2], [5,1,1,6], [5,2,0,6]])
    print r
    print 'RREF:'
    print r.rref()
    print 'Determinant:', r.det()
