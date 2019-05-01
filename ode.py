import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

#Physics parameters
g = 9.81
mu = 0.6
L = 1

#Integration interval in sec
delta_t = 0.001

#Time to simulate in sec
t_end = 10

#Initial Conditions
theta_0 = 0
theta_dot_0 = 2.8*np.pi

# Animation
fps = 25

###########################################################
#Solving damped pendulum ODE using simple euler integration
###########################################################

def theta_dot_dot(x,x_dot):
    return -mu * x_dot - g / L * np.sin(x)

theta = [theta_0]
theta_dot = [theta_dot_0]
t_grid = np.arange(0, t_end, delta_t)

for _ in t_grid[1:]:
    theta_dot.append(theta_dot[-1] + delta_t * theta_dot_dot(theta[-1],theta_dot[-1])) 
    theta.append(theta[-1] + delta_t * theta_dot[-1])

###################################################################
# Pendulum acceleration at various initial positions and velocities
###################################################################

theta_0_mesh, theta_dot_0_mesh = np.meshgrid(np.arange(-3*np.pi, 3*np.pi, 0.1), np.arange(-3*np.pi, 3*np.pi, 0.1))
theta_dot_dot_mesh = theta_dot_dot(theta_0_mesh, theta_dot_0_mesh)

#####################
#Matplotlib Animation
#####################

fig = plt.figure(1, figsize=(8, 8))
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(212)
ax2 = fig.add_subplot(222)

strm = ax1.streamplot(theta_0_mesh, theta_dot_0_mesh, theta_dot_0_mesh, theta_dot_dot_mesh, color=theta_dot_dot_mesh, cmap='coolwarm')
cbar = plt.colorbar(strm.lines, ax=ax1)
cbar.ax.get_yaxis().labelpad = -26
cbar.ax.set_ylabel('rad/$s^2$', rotation=270)
line0_1, = ax0.plot([], [])
line0_2, = ax0.plot([], [])
line1, = ax1.plot([], [], c='lime')
line2, = ax2.plot([], [], 'o-', lw=2)
line = [line0_1, line0_2, line1, line2]

time_template = 'time = %.1fs'
time_text = [ax.set_title('0') for ax in [ax0, ax1, ax2]]

ax0.set_xlabel('t [s]')
ax0.legend(['theta [rad]', 'theta_dot [rad/s]'],loc=4)
ax0.axis([0,t_end,np.min(theta + theta_dot),np.max(theta + theta_dot)])
ax1.set_xlabel('theta [rad]')
ax1.set_ylabel('theta_dot [rad/s]')
ax2.axis([-1.2,1.2,-1.2,1.2])

def init():
    for l in line:
        l.set_data([], [])
    for te in time_text:
        te.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, np.sin(theta[i])]
    thisy = [0, -np.cos(theta[i])]
    
    line[0].set_data(t_grid[:i+1], theta[:i+1])
    line[1].set_data(t_grid[:i+1], theta_dot[:i+1])
    line[2].set_data(theta[:i+1], theta_dot[:i+1])
    line[3].set_data(thisx, thisy)
    
    [tt.set_text(time_template % (t_grid[i])) for tt in time_text]
    
    return line, time_text


every_x_frame = 1 / (delta_t*fps) 
ani = animation.FuncAnimation(fig, animate, frames = np.arange(0,len(t_grid),int(every_x_frame)),
                              interval=1000/fps, blit=True, init_func=init, repeat=False)
ani.save('pendulum.mp4', fps=fps)

plt.show()