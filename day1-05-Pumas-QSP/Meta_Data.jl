# For the test simulation
defaults = [ y1 => 1.0,
                y2 => 0.0,
                y3 => 0.0,
                y4 => 0.0,
                y5 => 0.0,
                y6 => 0.0,
                y7 => 0.0,
                y8 => 0.0057,
                k1 => 1.71,
                k2 => 280.0,
                k3 => 8.32,
                k4 => 0.69,
                k5 => 0.43,
                k6 => 1.81]

# For the trials:
p_trial_1 = [ k6 => 20] 

p_trial_2 = [ k3 => 4.0,
                k4 => 0.4,
                k6 => 8.0] 
tspan_2 = (0.0, 20.0)

p_trial_3 = [ k3 => 15.0,
                k4 => 0.6,
                k6 => 2.0] 
tspan_3 = (0.0, 10.0)

# Population diversity window
search space = [k1 => (0. , 5.), k2 => (200., 300.), y1 => (0. , 3.)]