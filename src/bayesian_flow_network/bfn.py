from neural_networks.mlp import MLP
import torch

class BFN:
    def __init__(
            self,
            dim_in,
            dim_out,
            t_min=1e-9,
            x_min=-1,
            x_max=1,
            sigma_1=0.1,
            hidden=[200,200],
            dropout=0.1,
            activation="relu",
            cuda=False,
            seed=42,
            n=100):

        self.t_min = t_min
        self.x_min = x_min
        self.x_max = x_max
        self.sigma_1 = sigma_1
        self.dim_d_input = dim_in + 1
        self.dim_d_output = dim_out
        self.n = n

        self.model = MLP(
            dim_in=self.dim_d_input,
            hidden=hidden,
            dim_out=self.dim_d_output,
            dropout=dropout,
            activation=activation,
            cuda=cuda,
            seed=seed
        )

    def forward(self, x, y=None, data="continuous", time="discrete"):
        if data == "continuous" and time == "discrete":
            loss = self.forward_ctn_dsc(x, y)
        elif data == "continuous" and time == "continuous":
            loss = self.forward_ctn_ctn(x)
        else:
            raise NotImplementedError
        return loss

    def forward_ctn_dsc(self, x, y):
        n = self.n
        i = y
        t = i/n
        t = t.view(-1, 1)
        mu_sample, t, gamma = self._sender_ctn(x, t)
        x_hat = self._predict_ctn(mu_sample, t, gamma)
        loss = self.loss_ctn_dsc(x, x_hat, i, n)
        return loss

    def forward_ctn_ctn(self, x):
        t = torch.distributions.Uniform(
            low=torch.zeros(size=(x.shape[0],)),
            high=torch.ones(size=(x.shape[0],))).sample()
        t = t.view(-1, 1)
        mu_sample, t, gamma = self._sender_ctn(x, t)
        x_hat = self._predict_ctn(mu_sample, t, gamma)
        loss = self._loss_ctn_ctn(x, x_hat, t)
        return loss

    def _sender_ctn(self, x, t):
        # 1. CREATE SENDER DISTRIBUTION AND SAMPLE FROM IT
        
        # Eq. 73 states that γ(t)=β(t)/1+β(t)
        # Eq. 72 states that β(t)=σ_1^(−2t)−1
        # Thus the noise of the bayesian flow distribution is
        # eq.80 γ(t)=1-σ_1^(−2t)
        gamma = 1 - self.sigma_1**(2*t)

        # Add Noise to data sample
        mean = gamma * x
        # Generate standard normal random variables
        std_normal_samples = torch.randn_like(mean)
        # Create covariance matrix diagonal (variances)
        cov_diag = gamma * (1 - gamma)
        # Scale and shift the samples
        mu_sample = torch.sqrt(cov_diag) * std_normal_samples + mean

        return mu_sample, t, gamma

    def _predict_ctn(self, mu_sample, t, gamma):
        # 2. CREATE OUTPUT DISTRIBUTION

        # predict noise with network (i.e., add noise to x)
        input = torch.cat((mu_sample, t), dim=-1)
        e_hat = self.model(input)
        # calculate x_hat from mu_sample and e_hat
        x_hat = (mu_sample/gamma) - torch.sqrt((1-gamma)/gamma) * e_hat
        # clip x_hat
        x_hat = torch.clamp(x_hat, self.x_min, self.x_max)
        # zeros where t < t_min
        mask = t < self.t_min
        mask_expanded = mask.view(-1, 1).expand_as(x_hat)
        x_hat[mask_expanded] = 0
        return x_hat     
    
    def _loss_ctn_ctn(self, x, x_hat, t):
        # 3. RECEIVER DISTRIBUTION

        # Eq.88: The receiver distribution can be calculated using
        # x_hat and the variance of the sender distribution
        # It does not need to be calculated here as it is included in the loss

        # 4. CALCULATE LOSS
        # Proposition 3.1.:
        # The convolution of any continous distribution P with a
        # normal distribution N(0,σ^2) is a normal distribution N(E[P],σ^2)
        # Eq.41 (applying proposition 3.1.):
        # For continuous time, the KL divergence between sender distribution
        # and receiver distribution simplifies to the squared norm between x and x_hat
        # which are scaled by α(t) 
        # Eq.74: 
        # α(t)=−2ln(σ_1)/σ_1^2t
        # Eq. 101:
        # Applying Eq. 41 to Eq. 74 results in the loss function
        # for continuous time and continuous data
        squared_norm = torch.norm(x - x_hat, p=2)**2
        sigma1 = torch.tensor([self.sigma_1], dtype=torch.float)
        sigma1_2t = sigma1 ** (-2 * t)
        loss = -torch.log((sigma1 / sigma1_2t) * t * squared_norm)
        return loss.mean()
    
    def loss_ctn_dsc(self, x, x_hat, i, n):
        squared_norm = torch.norm(x - x_hat, p=2)**2
        sigma1 = torch.tensor([self.sigma_1], dtype=torch.float)
        numerator = n * (1-(sigma1 ** (2 / n)))
        denominator = 2 * (sigma1 ** ((2 * i) / n))
        loss = (numerator / denominator) * squared_norm
        return loss.mean()
    
    def generate_continuous(self, steps=10):
        data = []
        mu = torch.zeros(self.dim_d_input-1)
        rho = torch.ones(self.dim_d_input-1)
        for i in range(steps):
            t = (i+1)/steps
            gamma=torch.tensor([1-self.sigma_1**(2*t)])
            # predict noise with network (i.e., add noise to x)
            input = torch.cat((mu, torch.tensor([t])), dim=-1)
            e_hat = self.model(input)
            # calculate x_hat from mu_sample and e_hat
            x_hat = (mu/gamma) - torch.sqrt((1-gamma)/gamma) * e_hat
            # clip x_hat
            x_hat = torch.clamp(x_hat, self.x_min, self.x_max)
            # zeros where t < t_min
            if t < self.t_min:
                x_hat = torch.zeros(self.dim_d_input-1)
            alpha = self.sigma_1**((-2*(i+1))/steps)
            covariance_matrix = (1 / alpha) * torch.eye(len(x_hat))
            distribution = torch.distributions.MultivariateNormal(x_hat, covariance_matrix)
            y_sample = distribution.sample()
            mu = (rho*mu + alpha*y_sample) / (rho + alpha)
            rho = rho + alpha
            data.append(x_hat.detach().numpy())
        t = 1
        gamma=torch.tensor([1-self.sigma_1**(2*t)])
        # predict noise with network (i.e., add noise to x)
        input = torch.cat((mu, torch.tensor([t])), dim=-1)
        e_hat = self.model(input)
        # calculate x_hat from mu_sample and e_hat
        x_hat = (mu/gamma) - torch.sqrt((1-gamma)/gamma) * e_hat
        # clip x_hat
        x_hat = torch.clamp(x_hat, self.x_min, self.x_max)
        # zeros where t < t_min
        if t < self.t_min:
            x_hat = torch.zeros(self.dim_d_input-1)
        data.append(x_hat.detach().numpy())
        return data