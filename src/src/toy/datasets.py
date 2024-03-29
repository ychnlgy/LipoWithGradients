import torch, math, random

VISUALIZE_POINTS = 100
CONSTANT_SLICE = 0.5
TRUE_MASK = None

def create_from_str(dataname, N, D, noise):
    dataset = search(dataname)
    return create_from_dataset(dataset, N, D, noise)

def create_from_dataset(dataset, N, D, noise):
    
    T = dataset.get_num_true_features()

    if D == 0:
        D = T

    assert D >= T

    D_cols = shuffle_data_cols(D, T)

    global TRUE_MASK
    TRUE_MASK = [0] + D_cols
    
    X_data = blank(N*2, D)
    Y_data = dataset.calc_Y(X_data, D_cols)
    
    X_view = visual(X_data, D_cols, VISUALIZE_POINTS)
    Y_view = dataset.calc_Y(X_view, D_cols)

    return (

        (
            add_noise(X_data[:N], noise),
            add_noise(Y_data[:N], noise)
        ),

        (
            X_data[N:],
            Y_data[N:]
        ),

        (
            X_view,
            Y_view
        )
    )

def get_all():
    return [
        DatasetLinear(),
        DatasetForce(),
        DatasetGravity(),
        DatasetKinematics(),
        DatasetPendulum(),
        DatasetArrhenius(),
        DatasetSigmoid(),
        DatasetBogo(),
        DatasetMany()
    ]

def search(dataset):
    datasets = get_all()
    
    fmap = {d.get_nickname(): d for d in datasets}

    if dataset not in fmap:
        options = "\n\t".join(sorted([d.get_nickname() for d in datasets]))
        raise SystemExit("Choose from the following datasets:\n\t%s\n" % options)

    return fmap[dataset]

# === PRIVATE ===
    
def shuffle_data_cols(D, T):
    D_cols = list(range(1, D))
    random.shuffle(D_cols)
    return D_cols[:T-1]

def visual(X, D_cols, n_points):
    X_view = blank(n_points, X.size(-1))
    for i in D_cols:
        X_view[:,i] = CONSTANT_SLICE
    X_view[:,0] = torch.linspace(-1, 1, n_points)
    return X_view

def add_noise(M, noise):
    return M + torch.zeros_like(M).normal_(mean=0, std=noise)

def blank(N, D):
    return torch.rand(N, D)*2-1

class Dataset:

    def get_nickname(self):
        raise NotImplementedError

    def get_num_true_features(self):
        raise NotImplementedError

    def calc_Y(self, X, D_cols):
        return self.determine_output(X[:,0], *[X[:,i] for i in D_cols])

    def determine_output(self, *args):
        raise NotImplementedError

    def get_equation(self):
        return self.__doc__

class DatasetMany(Dataset):
    "Sum of 28 features."

    def get_nickname(self):
        return "many"

    def get_num_true_features(self):
        return 28

    def determine_output(self, *args):
        return sum(args)

class DatasetBogo(Dataset):
    "f(X) = x*p | x < N or (x+N/2)*p*discount | x >= N"

    def get_nickname(self):
        return "bogo"

    def get_num_true_features(self):
        return 4

    def determine_output(self, x, p, N, discount):
        p = p * 4
        N = N - 0.75
        out = x*p
        i = x>=N
        out[i] = ((out-N/2)*discount*0.1)[i]
        return out

class DatasetLinear(Dataset):
    "f(x) = 2a + b - 0.1x"

    def get_nickname(self):
        return "linear"

    def get_num_true_features(self):
        return 3

    def determine_output(self, x, a, b):
        return 2*a + b -0.1*x

class DatasetSigmoid(Dataset):
    "s(x) = c/(1+e^(-k(x-x0))) + y0"

    def get_nickname(self):
        return "sigmoid"

    def get_num_true_features(self):
        return 5

    def determine_output(self, x, c, k, x0, y0):
        return 2*c/(1+torch.exp(-k*10*(x-x0+0.5))) + y0 - 0.5

class DatasetGravity(Dataset):
    "G(r) = G*m1*m2/r^2"

    def get_nickname(self):
        return "gravity"

    def get_num_true_features(self):
        return 4

    def determine_output(self, r, G, m1, m2):
        EPS = 0.1
        return G*m1*m2/(r**2+EPS)

class DatasetForce(Dataset):
    "F(m) = m*a"

    def get_nickname(self):
        return "force"

    def get_num_true_features(self):
        return 2

    def determine_output(self, m, a):
        return m*a

class DatasetKinematics(Dataset):
    "d(t) = t*v0 + 0.5*a*t^2"

    def get_nickname(self):
        return "kinematics"

    def get_num_true_features(self):
        return 3

    def determine_output(self, t, v, a):
        return t*v + 0.5*a*t**2

class DatasetPendulum(Dataset):
    "f(t) = -g/l*sin(PI*t)"

    def get_nickname(self):
        return "pendulum"

    def get_num_true_features(self):
        return 3

    def determine_output(self, t, g, l_inv):
        return -g*l_inv*torch.sin(math.pi*t)

class DatasetArrhenius(Dataset):
    "k(T) = A*e^(-Ea*T/R)"

    def get_nickname(self):
        return "arrhenius"

    def get_num_true_features(self):
        return 3

    def determine_output(self, T, A, Ea):
        return A*2*torch.exp(-Ea*T*4)
