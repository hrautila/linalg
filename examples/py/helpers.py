
# some helpers to print matrix strings.

def str2(m, fmt='%7.2e'):
    s = ''
    for i in xrange(m.size[0]):
        s += "["
        for j in xrange(m.size[1]):
            if j != 0:
                s += ", "
            s += fmt % m[i, j]
        s += "]\n"
    return s


def gomat(m, fmt="%.2f", colmajor=True):
    """
    Print out matrix data as GO 2-dimensional array string that may pasted to
    a GO source file. Default ordering is column-major order. Setting 'colmajor'
    to False produces row major ordering of data structure.
    """
    s = '[][]float64{\n'
    N = m.size[1]
    if not colmajor: N = m.size[0]

    K = m.size[0]
    if not colmajor: K = m.size[1]

    for i in xrange(N):
        s += "[]float64{"
        for j in xrange(K):
            if j != 0:
                s += ", "
            if colmajor:
                s += fmt % m[j, i]
            else:
                s += fmt % m[i, j]
        if i != N-1:
            s += "},\n"
        else:
            s += "}}"
    return s


def strSpe(m, fmt="%.17f"):
    """
    Print out matrix data in string format that is parseable by matrix.FloatParseSpe()
    """
    s = ''
    for i in xrange(len(m)):
        if i != 0:
            s += ', '
        s += fmt % m[i]

    return "{%d %d [%s]}" % (m.size[0], m.size[1], s)


def printW(W):
    """
    Print out scaling matrix set.
    """
    for k in W.keys():
        if k == 'beta':
            print "** beta **\n", W[k]
        elif isinstance(W[k], list):
            for n in range(len(W[k])):
                print "** %s[%d] **\n" %(k, n), str2(W[k][n], "%.3f")
        else:
            print "** %s[0] **\n" % k, str2(W[k], "%.3f")


def run_go_test(name, refvals):
    import subprocess
    args = [name]
    for key in refvals:
        args += [ "-"+key, strSpe(refvals[key])]

    subprocess.call(args)

