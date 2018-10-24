using JuMP
using Ipopt
#using NLopt
using PyPlot
using FastGaussQuadrature

#目标航向在1～2.4之间时能够生成曲线
q0 = (0, 0, 0) #initial configuration
qf = (40.0, 40.0, 3.14) # final configuration
N = 9
@time weight = gausslegendre(N)
node = weight[1]   # interpolation node
coe = weight[2]    # interpolation coefficient
k0 = 0.            # initial curvature
k_max = 0.2        # max curvature
s_len_max = 2* hypot(qf[1]-q0[1], qf[2]-q0[2])  # max path length limit
lb = [-k_max, -k_max, -k_max, -k_max, 0]        #lower bound
ub = [k_max, k_max, k_max, k_max, s_len_max]    # upper bound

mdl = Model(solver = IpoptSolver())  # set model
#mdl = Model(solver=NLoptSolver(algorithm=:LD_SLSQP))

@variable(mdl, p[i=1:5], lowerbound=lb[i], upperbound=ub[i]) # set variable
initial_guess = [0.02, 0.016, 0.016, 0.017, 95.8]
for i=1:5
    setvalue(p[i], initial_guess[i])
end
# kppa(s) = k0 + bs + cs^2 + ds^3 + es^4 + fs^5
# theta(s) = q0[3] + k0s + b/2*s^2 + c/3*s^3 + d/4*s^4 + e/5*s^5 + f/6*s^6
@NLexpression(mdl, sf, p[5])
@NLexpression(mdl, b, (-137k0+300p[1]-300p[2]+200p[3]-75p[4])/(12sf))
@NLexpression(mdl, c, 25(45k0-154p[1]+214p[2]-156p[3]+61p[4])/(24sf^2))
@NLexpression(mdl, d, -125(17k0-71p[1]+118p[2]-98p[3]+41p[4])/(24sf^3))
@NLexpression(mdl, e, 625(3k0-14p[1]+26p[2]-24p[3]+11p[4])/(24sf^4))
@NLexpression(mdl, f, -625(k0-5p[1]+10p[2]-10p[3]+5p[4])/(24sf^5))
# Gauss-Legendre integration
@NLexpression(mdl, s[i=1:N], sf/2*(node[i]+1))
@NLexpression(mdl, expr_x[i=1:N], cos(q0[3] + k0*s[i] + b/2*s[i]^2 + c/3*s[i]^3 + d/4*s[i]^4 + e/5*s[i]^5 + f/6*s[i]^6)*sf/2)
@NLexpression(mdl, expr_y[i=1:N], sin(q0[3] + k0*s[i] + b/2*s[i]^2 + c/3*s[i]^3 + d/4*s[i]^4 + e/5*s[i]^5 + f/6*s[i]^6)*sf/2)
# target heading constraint:
@NLconstraint(mdl, q0[3] + k0*sf + b/2*sf^2 + c/3*sf^3 + d/4*sf^4 + e/5*sf^5 + f/6*sf^6 == qf[3])
# target x constraint:
@NLconstraint(mdl, q0[1] + sum(expr_x[i]*coe[i] for i=1:N) == qf[1])
#target y constraint:
@NLconstraint(mdl, q0[2] + sum(expr_y[i]*coe[i] for i=1:N) == qf[2])

#objective
@NLobjective(mdl,
    Min,
    k0^2*sf + k0*b*sf^2 +
    (b^2+2k0*c)/3*sf^3 +
    (2b*c+2k0*d)/4*sf^4 +
    (2b*d+c^2+2k0*e)/5*sf^5 +
    (2b*e+2c*d+2k0*f)/6*sf^6 +
    (2b*f+2c*e+d^2)/7*sf^7 +
    (2c*f+2d*e)/8*sf^8 +
    (2d*f+e^2)/9*sf^9 +
    (e*f)/5*sf^10 +
    f^2/11*sf^11)
# @NLobjective(mdl,Min,1)

@time status = solve(mdl)

p_val = getvalue(p)
b_val = getvalue(b)
c_val = getvalue(c)
d_val = getvalue(d)
e_val = getvalue(e)
f_val = getvalue(f)
println("p: ", p_val,
        "\nb: ", b_val,
        "\nc: ", c_val,
        "\nd: ", d_val,
        "\ne: ", e_val,
        "\nf: ", f_val)

# plot
function func_x(t, s1)
    s::Float64 =  (s1)/2 + (s1)*t/2
    return cos(q0[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4) + e_val/5*(s^5) + f_val/6*(s^6))*s1/2
end
function func_y(t, s1)
    s::Float64 =  (s1)/2 + (s1)*t/2
    return sin(q0[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4) + e_val/5*(s^5) + f_val/6*(s^6))*s1/2
end

n = Int64(ceil(p_val[5]))
h = p_val[5]/n
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


s_val = 0:h:p_val[5]
ks = b_val*s_val + c_val*s_val.^2 + d_val*s_val.^3 + e_val*s_val.^4 + f_val*s_val.^5

kfig = figure("kppa")
kx = kfig[:add_subplot](1,1,1)
kx[:plot](s_val, ks)

tha = b_val/2*(s_val.^2) + c_val/3*(s_val.^3) + d_val/4*(s_val.^4) + e_val/5*s_val.^5 + f_val/6*s_val.^6

tfig = figure("theta")
tx = tfig[:add_subplot](1,1,1)
tx[:plot](s_val, tha)
