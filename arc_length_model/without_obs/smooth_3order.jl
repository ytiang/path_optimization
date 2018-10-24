using JuMP
using Ipopt
using PyPlot
using FastGaussQuadrature

#目标航向在1～2.4之间时能够生成曲线
q0 = (0, 0, 0)
qf = (40.0, 40.0, 0.7)
N = 9
@time weight = gausslegendre(N)
node = weight[1]
coe = weight[2]
@show weight
k0 = 0.
k_max = 0.2
s_len_max = 3* hypot(qf[1]-q0[1], qf[2]-q0[2])
lb = [-k_max, -k_max, 0]
ub = [k_max, k_max, s_len_max]

mdl = Model(solver = IpoptSolver())

@variable(mdl, p[i=1:3], lowerbound=lb[i], upperbound=ub[i])
initial_guess = [0.039, -0.031, 97.03]
for i=1:3
    setvalue(p[i], initial_guess[i])
end

@NLexpression(mdl, b, -(11*k0 - 18*p[1] + 9*p[2])/(2*p[3]))
@NLexpression(mdl, c, (18*k0 - 45*p[1] + 36*p[2])/(2*p[3]^2))
@NLexpression(mdl, d, (-9*k0 + 27*p[1] - 27*p[2])/(2*p[3]^3))
@NLexpression(mdl, s[i=1:N], p[3]/2*(node[i]+1))
@NLexpression(mdl, expr_x[i=1:N], cos(q0[3] + k0*s[i] + b/2*s[i]^2 + c/3*s[i]^3 + d/4*s[i]^4)*p[3]/2)
@NLexpression(mdl, expr_y[i=1:N], sin(q0[3] + k0*s[i] + b/2*s[i]^2 + c/3*s[i]^3 + d/4*s[i]^4)*p[3]/2)
# target heading constraint:
@NLconstraint(mdl, q0[3] + k0*p[3] + b/2*p[3]^2 + c/3*p[3]^3 + d/4*p[3]^4 == qf[3])
# target x constraint:
 @NLconstraint(mdl, q0[1] + sum(expr_x[i]*coe[i] for i=1:N) == qf[1])
#target y constraint:
 @NLconstraint(mdl, q0[2] + sum(expr_y[i]*coe[i] for i=1:N) == qf[2])

#objective
@NLobjective(mdl,
    Min,
    (b^2 )/3*(p[3]^3) +
    (b*c)/2*(p[3]^4) +
    (2*b*d + c^2)/5*(p[3]^5) +
    c*d/3*(p[3]^6) +
    (d^2)/7*(p[3]^7))

@time status = solve(mdl)

p_val = getvalue(p)
b_val = getvalue(b)
c_val = getvalue(c)
d_val = getvalue(d)
println("p: ", p_val,
        "\nb: ", b_val,
        "\nc: ", c_val,
        "\nd: ", d_val)


function func_x(t, s1)
    local s =  (s1)/2 + (s1)*t/2
    return cos(q0[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4))*s1/2
end
function func_y(t, s1)
    local s =  (s1)/2 + (s1)*t/2
    return sin(q0[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4))*s1/2
end

n = Int64(ceil(p_val[3]))
h = p_val[3]/n
X = zeros(n+1)
Y = zeros(n+1)
X[1] = q0[1]
Y[1] = q0[2]
for i = 2 : n + 1
    si = (i-1) * h
    deltax = 0.
    deltay = 0.
    for k = 1:N
        deltax += func_x(node[k], si)*coe[k]
        deltay += func_y(node[k], si)*coe[k]
    end
    j::Int64 = i
    X[j] = q0[1] + deltax
    Y[j] = q0[1] + deltay
end

rfig = figure("path")
rx = rfig[:add_subplot](1,1,1)
rx[:plot](X, Y)


s_val = 0:h:p_val[3]
ks = b_val*s_val + c_val*s_val.^2 + d_val*s_val.^3

kfig = figure("kppa")
kx = kfig[:add_subplot](1,1,1)
kx[:plot](s_val, ks)

tha = b_val/2*(s_val.^2) + c_val/3*(s_val.^3) + d_val/4*(s_val.^4)

tfig = figure("theta")
tx = tfig[:add_subplot](1,1,1)
tx[:plot](s_val, tha)
