using JuMP
using Ipopt
# using NLopt
using PyPlot
using FastGaussQuadrature


τₖ = [0, 0.25, 0.5, 0.75, 1]
N = 15
num = N*(length(τₖ)-2) + (N+1)
df = [ [-25/3, 16, -12, 16/3, -1],
       [-1, -10/3, 6, -2, 1/3],
       [1/3, -8/3, 0, 8/3, -1/3],
       [-1/3, 2, -6, 10/3, 1],
       [1, -16/3, 12, -16, 25/3]]
intf² = [ [0.0, 0.0, 0.0, 0.0, 0.0],
         [74023/1451520, 5843/25920, 103/2520, 173/25920, 313/1451520],
         [2323/45360, 124/405, 52/315, 4/405, 13/45360],
         [919/17920, 99/320, 81/280, 29/320, 9/17920],
         [146/2835, 128/405, 104/315, 128/405, 146/2835]]
vel = 10
L = 2.54
b = 0.8
lf = 0.35
lr = 0.4
x_obs = 20.0
y_obs = 20.0
r_obs = 8.0

#目标航向在1～2.4之间时能够生成曲线
q0 = (0, 0, 0) #initial configuration
qf = (40.0, 40.0, 0.0) # final configuration

mdl = Model(solver = IpoptSolver())  # set model
 # mdl = Model(solver = NLoptSolver(algorithm = :LD_SLSQP))
@variable(mdl, x[i=1:num])
@variable(mdl, y[i=1:num])
@variable(mdl, -pi <= θ[i=1:num] <= pi)     # heading
@variable(mdl, -0.44 <= u[i=1:num] <= 0.44) # steering control bound
@variable(mdl, 0.1 <= tf <= 100.0)
# heading = atan2(qf[2]-q0[2], qf[1]-q0[1])
# path_length = hypot(qf[1]-q0[1], qf[2]-q0[2])
# for i=1:num
#     setvalue(x[i], path_length/num*i*cos(heading))
#     setvalue(y[i], path_length/num*i*sin(heading))
#     setvalue(θ[i], heading)
# end
@NLexpression(mdl, h, tf/N)
# state equation constraint
# dx = v*cos(θ)
# dy = v*sin(θ)
# dθ = v*tan(δ)/L
# u = δ
for i = 1:N
    id_start::Int32 = (length(τₖ)-1)*(i-1) + 1
    for j = 1:5
        @NLconstraint(mdl, sum(x[id_start+k-1]*df[j][k] for k=1:5) == h*vel*cos(θ[id_start+j-1]))
        @NLconstraint(mdl, sum(y[id_start+k-1]*df[j][k] for k=1:5) == h*vel*sin(θ[id_start+j-1]))
        @NLconstraint(mdl, sum(θ[id_start+k-1]*df[j][k] for k=1:5) == h*vel*tan(u[id_start+j-1])/L)
    end
end
# bounds constraint
@constraint(mdl, x[1] == q0[1])
@constraint(mdl, y[1] == q0[2])
@constraint(mdl, θ[1] == q0[3])
@constraint(mdl, x[num] == qf[1])
@constraint(mdl, y[num] == qf[2])
@constraint(mdl, θ[num] == qf[3])
#enviroment constraint
# for i = 1:num
#     @NLconstraint(mdl, (x[i]-x_obs)^2 + (y[i]-y_obs)^2 >= r_obs^2)
# end
# objective
# minium time and minium control
@NLobjective(mdl,
    Min,
    0.2*tf + h*sum(u[4*(i-1)+1]^2*(intf²[5][1]-intf²[1][1]) +
          u[4*(i-1)+2]^2*(intf²[5][2]-intf²[1][2]) +
          u[4*(i-1)+3]^2*(intf²[5][3]-intf²[1][3]) +
          u[4*(i-1)+4]^2*(intf²[5][4]-intf²[1][4]) +
          u[4*(i-1)+5]^2*(intf²[5][5]-intf²[1][5]) for i=1:N))

@time status = solve(mdl)
X = getvalue(x)
Y = getvalue(y)
t = getvalue(tf)
print("tf: ",t,"\n")
#
# # plot
rfig = figure("path")
rx = rfig[:add_subplot](1,1,1)
rx[:plot](X, Y)
θo = -3.1:0.1:3.14
xo = x_obs*ones(length(θo)) + r_obs*cos(θo)
yo = y_obs*ones(length(θo)) + r_obs*sin(θo)
rx[:plot](xo, yo)
#
#
# s_val = 0:h:p_val[5]
# ks = b_val*s_val + c_val*s_val.^2 + d_val*s_val.^3 + e_val*s_val.^4 + f_val*s_val.^5
#
# kfig = figure("kppa")
# kx = kfig[:add_subplot](1,1,1)
# kx[:plot](s_val, ks)
#
# tha = b_val/2*(s_val.^2) + c_val/3*(s_val.^3) + d_val/4*(s_val.^4) + e_val/5*s_val.^5 + f_val/6*s_val.^6
#
# tfig = figure("theta")
# tx = tfig[:add_subplot](1,1,1)
# tx[:plot](s_val, tha)
