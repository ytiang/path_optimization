using JuMP
using Ipopt
# using NLopt
using PyPlot
using FastGaussQuadrature
N = 10 # number of curve segment, (N+1) collocation points
degree = 3 # # each segment use a lagrange polynomial to approximate
collocation_pt = gaussradau(N) # the domain of gauss-radau is:[-1,1)
knot = collocation_pt[1]
push!(knot, 1.0)
lagrange_pt = gaussradau(degree)
print("knot: ",knot,"\n")
# print(lagrange_pt)

num = (N+1)+(degree-1)*N
vel = 10
L = 2.54
b = 0.8
lf = 0.35
lr = 0.4
x_obs = [20.0, 20.0]
y_obs = [15.0, 30.0]
r_obs = [5.0, 5.0]

q0 = (0, 0, 0) #initial configuration
qf = (40.0, 40.0, 0.0) # final configuration
# dl = [[-6.499997 -1.404989 0.328531 -0.137169 0.066853 -0.100000],
#       [9.109642 -0.290622 -1.259161 0.432951 -0.199852 0.296001],
#       [-4.388562 2.594172 -0.428380 -1.181899 0.437592 -0.626272],
#       [3.485129 -1.696575 2.248006 -0.903037 -1.445966 1.820731],
#       [-4.206223 1.939335 -2.061085 3.580697 -4.377995 -13.890491],
#       [2.500011 -1.141322 1.172090 -1.791542 5.519368 12.500030]]
dl = [[-2.500000 -0.526599 0.126599 -0.166667],
      [3.765986 -0.387627 -0.583920 0.691071],
      [-2.765986 1.783920 -1.612373 -5.024405],
      [1.500000 -0.869694 2.069694 4.500001]]

xlb = -10.0*ones(num,1)
xub = 70.0*ones(num,1)
ylb = -10.0*ones(num,1)
yub = 70.0*ones(num,1)
θlb = -pi*ones(num,1)
θub = pi*ones(num,1)
xlb[1] = q0[1]
xub[1] = q0[1]
ylb[1] = q0[2]
yub[1] = q0[2]
θlb[1] = q0[3]
θub[1] = q0[3]
xlb[num] = qf[1]
xub[num] = qf[1]
ylb[num] = qf[2]
yub[num] = qf[2]
θlb[num] = qf[3]
θub[num] = qf[3]


mdl = Model(solver = IpoptSolver())  # set model

@variable(mdl, x[i=1:num],lowerbound=xlb[i], upperbound=xub[i])
@variable(mdl, y[i=1:num],lowerbound=ylb[i], upperbound=yub[i])
@variable(mdl, θ[i=1:num],lowerbound=θlb[i], upperbound=θub[i])     # heading
@variable(mdl, -0.44 <= u[i=1:num] <= 0.44) # steering control bound
@variable(mdl, 0.1<= tf <= 50.0)
heading = atan2(qf[2]-q0[2], qf[1]-q0[1])
path_length = hypot(qf[1]-q0[1], qf[2]-q0[2])
for i=1:num
    setvalue(x[i], path_length/num*(i-1)*cos(heading))
    setvalue(y[i], path_length/num*(i-1)*sin(heading))
    setvalue(θ[i], heading)
end
@NLexpression(mdl, Ax[i=1:num], x[i] + (L+lf)*cos(θ[i]) - b*sin(θ[i]))
@NLexpression(mdl, Ay[i=1:num], y[i] + (L+lf)*sin(θ[i]) + b*cos(θ[i]))
@NLexpression(mdl, Bx[i=1:num], x[i] + (L+lf)*cos(θ[i]) + b*sin(θ[i]))
@NLexpression(mdl, By[i=1:num], y[i] + (L+lf)*sin(θ[i]) - b*cos(θ[i]))
@NLexpression(mdl, Cx[i=1:num], x[i] - lr*cos(θ[i]) + b*sin(θ[i]))
@NLexpression(mdl, Cy[i=1:num], y[i] - lr*sin(θ[i]) - b*cos(θ[i]))
@NLexpression(mdl, Dx[i=1:num], x[i] - lr*cos(θ[i]) - b*sin(θ[i]))
@NLexpression(mdl, Dy[i=1:num], y[i] - lr*sin(θ[i]) + b*cos(θ[i]))
@NLexpression(mdl, tk[i=1:N+1], tf/2*(knot[i]+1))
for i = 1:N  # (N) segment
    id_start::Int32 = degree*(i-1)
    for j = 1:degree # t1, t2, t3
        @NLconstraint(mdl, 2/(tk[i+1]-tk[i])*sum(x[id_start+k]*dl[k][j] for k=1:(degree+1)) == vel*cos(θ[id_start+j]))
        @NLconstraint(mdl, 2/(tk[i+1]-tk[i])*sum(y[id_start+k]*dl[k][j] for k=1:(degree+1)) == vel*sin(θ[id_start+j]))
        @NLconstraint(mdl, 2/(tk[i+1]-tk[i])*sum(θ[id_start+k]*dl[k][j] for k=1:(degree+1)) == vel*tan(u[id_start+j])/L)
    end
end

#enviroment constraint
for i = 2:num-1
    # @NLconstraint(mdl, (x[i]-x_obs)^2 + (y[i]-y_obs)^2 >= r_obs^2)
    for j =1:length(x_obs)
        @NLconstraint(mdl, (Ax[i]-x_obs[j])^2 + (Ay[i]-y_obs[j])^2 >= r_obs[j]^2)
        @NLconstraint(mdl, (Bx[i]-x_obs[j])^2 + (By[i]-y_obs[j])^2 >= r_obs[j]^2)
        @NLconstraint(mdl, (Cx[i]-x_obs[j])^2 + (Cy[i]-y_obs[j])^2 >= r_obs[j]^2)
        @NLconstraint(mdl, (Dx[i]-x_obs[j])^2 + (Dy[i]-y_obs[j])^2 >= r_obs[j]^2)
    end
end
# objective
# minium time and minium control
@NLobjective(mdl,
    Min,
    0.9*tf + sum((tk[i+1]-tk[i])/2*sum(u[degree*(i-1)+j]^2*lagrange_pt[2][j] for j=1:degree) for i=1:N))


@time status = solve(mdl)
X = getvalue(x)
Y = getvalue(y)
t = getvalue(tf)
θ_v = getvalue(θ)
tk_v = getvalue(tk)
A1 = getvalue(Ax)
B1 = getvalue(Bx)
C1 = getvalue(Cx)
D1 = getvalue(Dx)
A2 = getvalue(Ay)
B2 = getvalue(By)
C2 = getvalue(Cy)
D2 = getvalue(Dy)
print("tf: ", t, "\n",
      "X: ", X, "\n",
      "Y: ", Y, "\n")

# # plot
function Lagrange_func(var, t, seg_id)
    l = zeros(degree+1)
    l[1] = (.2958758458*(-1.408248392*t-.4082483923))*(t-.689898)*(t-1.0)
    l[2] = (.7912413377*(1.408248392*t+1.408248392))*(t-.689898)*(t-1.0)
    l[3] = -(3.291241653*(.5917516915*t+.5917516915))*(t+.289898)*(t-1.0)
    l[4] = (2.500000315*(.5000000000*t+.5000000000))*(t+.289898)*(t-.689898)
    id_start::Int32 = degree*(seg_id-1)
    result = 0.0
    for i=1:degree+1
        result += l[i]*var[id_start+i]
    end
    return result
end
x_val = []
y_val = []
θ_val = []
s_val = [0.0]

for i=1:N
    for τ = -1.0:0.01:0.99
        push!(x_val, Lagrange_func(X, τ, i))
        push!(y_val, Lagrange_func(Y, τ, i))
        push!(θ_val, Lagrange_func(θ_v, τ, i))
        if length(x_val) > 1
            len = length(x_val)
            ∇s = hypot(y_val[len]-y_val[len-1], x_val[len]-x_val[len-1])
            push!(s_val, s_val[length(s_val)] + ∇s)
        end
    end
end
rfig = figure("path")
rx = rfig[:add_subplot](1,1,1)
rx[:scatter](X, Y)
rx[:plot](x_val, y_val)
for i=1:length(x_obs)
    θo = -3.1:0.1:3.14
    xo = x_obs[i]*ones(length(θo)) + r_obs[i]*cos(θo)
    yo = y_obs[i]*ones(length(θo)) + r_obs[i]*sin(θo)
    rx[:plot](xo, yo)
end
for i=1:num
    x = []
    y = []
    push!(x, A1[i])
    push!(x, B1[i])
    push!(x, C1[i])
    push!(x, D1[i])
    push!(x, A1[i])
    push!(y, A2[i])
    push!(y, B2[i])
    push!(y, C2[i])
    push!(y, D2[i])
    push!(y, A2[i])
    rx[:plot](x,y)
end
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
tfig = figure("theta")
tx = tfig[:add_subplot](1,1,1)
tx[:plot](s_val, θ_val)
