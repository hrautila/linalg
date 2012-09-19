
# some helpers to print matrix strings.

def str2(m, fmt='%7.2e', rowmajor=True):
    s = ''
    if isinstance(m, list):
        # cp epigraph thing
        s = str2(m[0], fmt, rowmajor)
        s += ("/"+fmt+"/\n") % m[1]
    else:
        if rowmajor:
            for i in xrange(m.size[0]):
                s += "["
                for j in xrange(m.size[1]):
                    if j != 0:
                        s += ", "
                    s += fmt % m[i, j]
                s += "]\n"
        else:
            for i in xrange(m.size[1]):
                s += "["
                for j in xrange(m.size[0]):
                    if j != 0:
                        s += ", "
                    s += fmt % m[j, i]
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


def printW(W, fmt="%.3f"):
    """
    Print out scaling matrix set.
    """
    from cvxopt import matrix

    for k in W.keys():
        if k == 'beta':
            print "** beta **\n", str2(matrix(W[k]), fmt)
        elif isinstance(W[k], list):
            for n in range(len(W[k])):
                print "** %s[%d] **\n" %(k, n), str2(W[k][n], fmt)
        else:
            print "** %s[0] **\n" % k, str2(W[k], fmt)


def run_go_test(name, refvals):
    import subprocess
    args = [name]
    for key in refvals:
        args += [ "-"+key, strSpe(refvals[key])]

    subprocess.call(args)



spvariables = {}
spmajor = 0
spminor = 0
sppath = '.'
spactive = False
spstack = []

def sp_reset(path):
    global spmajor, spminor, sppath, spvariables
    spvariables = {}
    spmajor = 0
    spminor = 0
    sppath = path
    spactive = False

def sp_activate():
    global spactive
    spactive = True

def sp_major():
    global spmajor
    return spmajor

def sp_major_next():
    if spactive:
        global spmajor, spminor
        spmajor += 1
        spminor = 0

def sp_minor_push(val):
    if spactive:
        global spstack
        spstack.append(val)

def sp_minor_pop():
    if spactive:
        global spstack
        minor = spstack[-1]
        del spstack[-1]
        return minor

def sp_minor_empty():
    global spstack
    return len(spstack) == 0

def sp_minor_top():
    if spactive:
        global spstack
        return spstack[-1]
    return 0

def sp_add_var(name, var):
    global spvariables
    spvariables[name] = var


def sp_create(name, minor, singletons={}):
    import os.path
    from cvxopt import matrix

    global spactive
    if not spactive:
        return

    path = os.path.join(sppath, "%04d-%04d." % (spmajor, minor) + name)
    #print "sp_create: path=", path
    try:
        with open(path, "w+") as fp:
            fp.write("name: "+name+"\n")
            for k, v in spvariables.items():
                if isinstance(v, matrix):
                    # normal matrix
                    fp.write("%s matrix 1: %s\n" % (k, strSpe(v)))
                elif isinstance(v, list):
                    # epigraph thing
                    fp.write("%s epigraph 1: [%.17f %s]\n" % (k, v[1], strSpe(v[0])))
                elif isinstance(v, dict):
                    # W scaling matrices
                    for key, data in v.items():
                        if key == 'beta':
                            fp.write("beta.0 matrix 1: %s\n" % strSpe(matrix(data)))
                        else:
                            if isinstance(data, list):
                                i = 0
                                for d in data:
                                    fp.write("%s.%d matrix 1: %s\n" % \
                                                 (key, i, strSpe(d)))
                                    i += 1
                            else:
                                 fp.write("%s.0 matrix 1: %s\n" %  (key, strSpe(data)))
                        #endif
                    #endfor
                #endif
            # write out singleton variables
            for varname, val in singletons.items():
                fp.write("%s float 1: %.17f\n" % (varname, val))

    except IOError, e:
        print "sp_create error: ", str(e)
        
                
def sp_create_next(name):
    global spactive
    #print "sp_create_next: spactive=", str(spactive)
    if spactive:
        sp_minor_next()
        sp_create(name)

