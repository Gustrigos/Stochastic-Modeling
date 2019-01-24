# Stochastic-Modeling
Utilizing three different stochastic processes to forecast equity prices. 

## Stochastic Process
A Stochastic process is any process that describes the evolution of a random phenomenon in time. According to the random walk hypothesis in financial theory, stock market prices evolve according to a random walk and thus cannot be predicted. With the efficient-market hypothesis being consistent with the random walk hypothesis, we can therefore try to model different stochastic processes to describe the movement of the SPY for a chosen number of time-steps. In this case, we are going to model three different stochastic processes, mainly used for asset pricing, but that can be extended to explain portfolio behavior and therefore be used as a risk management tool for case scenario analysis. The stochastic models are the following: Brownian motion (wiener process), geometric Brownian motion, and Jump Diffusion model.

# Models
For the models, we use Python with the scientific libraries of Numpy, SciPy, Matplotlib, and Pandas. 
We establish a predetermined set of parameters like the number of simulations we will do for each random process, the number of time intervals to be forecasted, and other statistical parameters gathered from the historical intraday prices of the SPY S&P 500 liquid ETF. (Logarithmic returns, CAGR, Volatility, etc.).

## Brownian Model
Brownian motion, originally used to model the random motion of liquid and gas particles, was introduced by Louis Bachelier in relation to the financial markets to evaluate stock options in his paper Theory of Speculation. Black and Scholes later emphasized some of the properties used by Bachelier to build the Black-Scholes Formula. In financial theory, Brownian motion is often described as a Wiener process, where the 0th step has a value of 0, the function of time minus the wiener process at that time is continuous, and the wiener process has independent normally distributed increments.

Brownian motion is describing that at a given time interval, the current price of the stock will be equal to the previous price multiplied by a random variable (Wiener process). With a statistically coherent number of samples, we will get a normally distributed set of prices with small standard deviations. 

## Geometric Brownian Model
To derive the Black Scholes equation, Fisher Black and Myron Scholes presented geometric Brownian motion as a model that contained the Wiener process Z with the addition of a drift module and volatility module. Where geometric Brownian motion explains that the change in an asset price at a predetermined time is equal to the percentage drift expected annually times the current price, plus the daily volatility expected annually in the asset price times the Wiener process. 

Geometric Brownian motion, with a statistically coherent number of samples, has a more robust, practical application to the movement of stock prices. The simulations will have a normal distribution bell shape but will have a much bigger standard deviation, which expands the maximum expected drawdown and maximum expected upside for the asset. This will include heavier tails but will still have a log-normal to normal shape. 

## Merton Jump Diffusion Model
Robert C. Merton addressed some of the limitations of the geometric Brownian motion presented by Black and Scholes. Because both the Brownian motion and geometric Brownian motion followed the normal distribution and didn't take into consideration the discontinuity occurring in the stochastic processes. Merton added a jump and diffusion component to the geometric Brownian motion formula, in which the model adds sudden jumps and discontinuities in the stochastic process to address highly-improbable events like stock crashes and therefore make the model more robust to tail risk by making the simulation results have heavier downside tails. An optional addition to this model is to add sudden upside jumps to explain the behavior of stocks with a more symmetric probability shape.

