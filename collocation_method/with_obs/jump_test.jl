#!/home/yangt/workspace/software/julia/bin/julia
# julia 的变量全是引用！
# 出现InExactError的原因可能有：使用变量未声明、声明时默认为整型，赋值时却为float型，超出了范围，比如0和0.的区别
using FastGaussQuadrature
using JuMP
using Ipopt
# using NLopt
using PyPlot
using Plotly
using CSV
FILE1 = "/home/yangt/catkin_ws/src/path_optimization/data/distance.csv"
FILE2 = "/home/yangt/catkin_ws/src/path_optimization/data/gradient.csv"

dist_file = CSV.read(FILE1; header = true)
grad_file = CSV.read(FILE2; header = true)
# map parameters:
stride = 800
column = 800
row = 800
x_pos = Array(dist_file[:x])
y_pos = Array(dist_file[:y])
distance_map = Array(dist_file[:dis])
diffx_map = Array(grad_file[:dx])
diffy_map = Array(grad_file[:dy])
resolution = 0.1
# plot(x_pos, y_pos, distance_map)
#坐标为(x,y)的点在数组中的位置为：
# intergration related
N = 9
@time weight = gausslegendre(N)
node = weight[1]
coe = weight[2]
# global variable
k0 = 0. # initial kppa
is_get_initial_state = 0
is_get_goal_state = 0
dis_threshold = 5

# init model
k_max = 0.2 # max kppa
s_len_max = 500.0
lb = [-k_max, -k_max, 0]
ub = [k_max, k_max, s_len_max]
# set model
# initial_guess = [0.0623997, -0.02136, 45.4845]
initial_guess = [0.0, 0.0, 45.4845]
initial_state = [5., 5., 0.]
goal_state = [40.0, 40.0, 0.7]
function mapAt(x, y, map)
    new_x = max(min(80.0, x), 0.0)
    new_y = max(min(80.0, y), 0.0)
    indexx = ceil(new_x/resolution)
    indexy = ceil(new_y/resolution)
    dist = 0.
    i::Int32 = row - indexx + 1
    j::Int32 = column - indexy
    index::Int32 = i+j*stride
    if index > 640000
        print("***************get: ", index, ", ",i ,", ", j,"; ","input: ",x,", ",y,"\n")
    end
    dist = map[index]
    return dist
end

function dist(x, y)
    d = mapAt(x, y, distance_map)
    return d
end
function ∇dist(g, x, y)
    g[1] = mapAt(x, y, diffx_map)
    g[2] = mapAt(x, y, diffy_map)
end

function pathPt(s)
    Δx = 0
    Δy = 0
    pt = [0., 0.]
    local q = getvalue(q0)
    local b_val = getvalue(b)
    local c_val = getvalue(c)
    local d_val = getvalue(d)
    for j = 1:N
        sⱼ = s/2*(node[j]+1)
        θⱼ = q[3] + k0*sⱼ + b_val/2*sⱼ^2 + c_val/3*sⱼ^3 + d_val/4*sⱼ^4
        Δx += cos(θⱼ) * coe[j] * s / 2
        Δy += sin(θⱼ) * coe[j] * s / 2
    end
    pt[1] = q[1] + Δx
    pt[2] = q[2] + Δy
    return pt
end
#objective
function myObject(p1, p2, p3)
    local b = -(11*k0 - 18*p1 + 9*p2)/(2*p3)
    local c = (18*k0 - 45*p1 + 36*p2)/(2*p3^2)
    local d = (-9*k0 + 27*p1 - 27*p2)/(2*p3^3)
    q = getvalue(q0)
    # obstacle term
    J₁ = 0.
    for k=1:N
        sₖ = p3/2*(node[k]+1)
        rₖ = pathPt(sₖ)
        # print("yyyyyyyyyyyyyyyyyyyyy: ",J₁,"\n")
        J₁ += mapAt(rₖ[1], rₖ[2], distance_map) * coe[k] * p3/2

    end
    # smooth term
    J₂ = 0.
    J₂ = (b^2 )/3*(p3^3) +
          (b*c)/2*(p3^4) +
          (2*b*d + c^2)/5*(p3^5) +
          c*d/3*(p3^6) +
          (d^2)/7*(p3^7)
    obj = J₁ #+ J₂
    print("************J1, J2, obj: ",J₁," ",J₂," ",obj,"\n")
    return obj
end
function ∇myObject(g, p1, p2, p3)
    local b = -(11*k0 - 18*p1 + 9*p2)/(2*p3)
    local c = (18*k0 - 45*p1 + 36*p2)/(2*p3^2)
    local d = (-9*k0 + 27*p1 - 27*p2)/(2*p3^3)
    g1 = [0.,0.,0.]
    g2 = [0.,0.,0.]
    g = [0.,0.,0.]
    ∂b∂p = [0.,0.,0.]
    ∂c∂p = [0.,0.,0.]
    ∂d∂p = [0.,0.,0.]
    ∂b∂p[1] = 9/p3
    ∂b∂p[2] = -9/(2*p3)
    ∂b∂p[3] =  (11*k0 - 18*p1 + 9*p2)/(2*p3^2)
    ∂c∂p[1] = -45/(2*p3^2)
    ∂c∂p[2] = 18/(p3^2)
    ∂c∂p[3] = -(18*k0 - 45*p1 + 36*p2)/(p3^3)
    ∂d∂p[1] = 27/(2*p3^3)
    ∂d∂p[2] = -27/(2*p3^3)
    ∂d∂p[3] = -3(-9*k0 + 27*p1 - 27*p2)/(2*p3^4)
    q = getvalue(q0)
    # # obstacle term gradient
    for k = 1 : N
        sₖ = p3/2*(node[k]+1)
        θₖ = q[3] + k0*sₖ +  b/2*sₖ^2 + c/3*sₖ^3 + d/4*sₖ^4
        κₖ = k0 + b*sₖ + c*sₖ^2 + d*sₖ^3
        Δxₖ = 0.
        Δyₖ = 0.
        ∂Xₖ∂p = [0.,0.,0.]
        ∂Yₖ∂p = [0.,0.,0.]
        for j = 1:N
            sₖⱼ = sₖ/2*(node[j]+1)
            θⱼ = q[3] + k0*sₖⱼ + b/2*sₖⱼ^2 + c/3*sₖⱼ^3 + d/4*sₖⱼ^4
            Δxₖ += cos(θⱼ) * coe[j] * sₖ / 2
            Δyₖ += sin(θⱼ) * coe[j] * sₖ / 2
            for i=1:3
                ∂Xₖ∂p[i] += -sin(θⱼ)* (1/2*∂b∂p[i]*sₖⱼ^2 + 1/3*∂c∂p[i]*sₖⱼ^3 + 1/4*∂d∂p[i]*sₖⱼ^4) * coe[j]*sₖ / 2
                ∂Yₖ∂p[i] += cos(θⱼ)* (1/2*∂b∂p[i]*sₖⱼ^2 + 1/3*∂c∂p[i]*sₖⱼ^3 + 1/4*∂d∂p[i]*sₖⱼ^4) * coe[j]*sₖ / 2
            end
        end
        xₖ = q[1] + Δxₖ
        yₖ = q[2] + Δyₖ
        # print("*************************xₖ, yₖ: ", xₖ," ",yₖ,"\n")
        # mapAt(xₖ,yₖ,distance_map)
        vₖ = mapAt(xₖ, yₖ, distance_map)
        # print("dis: ", vₖ,"\n")
        ∇vₖ = [0., 0.]
        ∇vₖ[1] = -mapAt(xₖ, yₖ, diffx_map)
        ∇vₖ[2] = -mapAt(xₖ, yₖ, diffy_map)
        # print("∇dis: ", ∇vₖ,"\n")
        ∂f∂xₖ = -sin(θₖ)^2 * ∇vₖ[1] - cos(θₖ)*sin(θₖ)*∇vₖ[2] + vₖ*κₖ*sin(θₖ)
        ∂f∂yₖ = -cos(θₖ)*sin(θₖ)*∇vₖ[1] + cos(θₖ)^2 *∇vₖ[2] - vₖ*κₖ*cos(θₖ)
        g1[1] += (∂f∂xₖ*∂Xₖ∂p[1] + ∂f∂yₖ*∂Yₖ∂p[1]) * coe[k] * p3/2
        g1[2] += (∂f∂xₖ*∂Xₖ∂p[2] + ∂f∂yₖ*∂Yₖ∂p[3]) * coe[k] * p3/2
        g1[3] += (∂f∂xₖ*∂Xₖ∂p[3] + ∂f∂yₖ*∂Yₖ∂p[3]) * coe[k] * p3/2
    end
    # print("g1: ",g1[1], " ,",g1[2]," ,",g1[3],"\n")
    # smooth term gradient
    g2[1] = -3//280 * p3 * (59 * k0 - 72 * p1 + 9 * p2)
    g2[2] = -3//280 * p3 * (74 * k0 + 9 * p1 - 72 * p2)
    g2[3] = 347//420 * k0 ^ 2 - 177//280 * k0 * p1 -
           111//140 * k0 * p2 + 27//70 * p1 ^ 2 -
           27//280 * p1 * p2 + 27//70 * p2 ^ 2
    # print("g2: ",g2[1], " ,",g2[2]," ,",g2[3],"\n")
    g[1] = g1[1] #+ g2[1]
    g[2] = g1[2] #+ g2[2]
    g[3] = g1[3] #+ g2[3]
    # print("∇J: ",g[1], " ,",g[2]," ,",g[3],"\n")
end
mdl = Model(solver = IpoptSolver())
# mdl = Model(solver = NLoptSolver(algorithm=:LD_SLSQP))
@variable(mdl, p[i=1:3], lowerbound=lb[i], upperbound=ub[i])
@NLparameter(mdl, q0[i=1:3] == initial_state[i])
@NLparameter(mdl, qf[i=1:3] == goal_state[i])
for i=1:3
    setvalue(p[i], initial_guess[i])
end
@NLexpression(mdl, b, -(11*k0 - 18*p[1] + 9*p[2])/(2*p[3]))
@NLexpression(mdl, c, (18*k0 - 45*p[1] + 36*p[2])/(2*p[3]^2))
@NLexpression(mdl, d, (-9*k0 + 27*p[1] - 27*p[2])/(2*p[3]^3))
@NLexpression(mdl, si[i=1:N], p[3]/2*(node[i]+1))
@NLexpression(mdl, expr_x[i=1:N], cos(q0[3] + k0*si[i] + b/2*si[i]^2 + c/3*si[i]^3 + d/4*si[i]^4)*p[3]/2)
@NLexpression(mdl, expr_y[i=1:N], sin(q0[3] + k0*si[i] + b/2*si[i]^2 + c/3*si[i]^3 + d/4*si[i]^4)*p[3]/2)
# target heading constraint:
@NLconstraint(mdl, q0[3] + k0*p[3] + b/2*p[3]^2 + c/3*p[3]^3 + d/4*p[3]^4 == qf[3])
# target x constraint:
@NLconstraint(mdl, q0[1] + sum(expr_x[i]*coe[i] for i=1:N) == qf[1])
#target y constraint:
@NLconstraint(mdl, q0[2] + sum(expr_y[i]*coe[i] for i=1:N) == qf[2])

# JuMP.register(mdl, :myObject, 3, myObject, autodiff = true)
JuMP.register(mdl, :myObject, 3, myObject, ∇myObject)
@NLobjective(mdl, Min, myObject(p[1], p[2], p[3]))
# @NLobjective(mdl, Min, 1)
# @NLobjective(mdl,
#     Min,
#     (b^2 )/3*(p[3]^3) +
#     (b*c)/2*(p[3]^4) +
#     (2*b*d + c^2)/5*(p[3]^5) +
#     c*d/3*(p[3]^6) +
#     (d^2)/7*(p[3]^7))

@time status = solve(mdl)
p_val = getvalue(p)
b_val = getvalue(b)
c_val = getvalue(c)
d_val = getvalue(d)
q0_val = getvalue(q0)
qf_val = getvalue(qf)
println("p: ", p_val,
        "\nb: ", b_val,
        "\nc: ", c_val,
        "\nd: ", d_val)


function func_x(t, s1)
    s::Float64 =  (s1)/2 + (s1)*t/2
    return cos(q0_val[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4))*s1/2
end
function func_y(t, s1)
    s::Float64 =  (s1)/2 + (s1)*t/2
    return sin(q0_val[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4))*s1/2
end

n = ceil(p_val[3])
h = p_val[3]/n
X = zeros(n+1)
Y = zeros(n+1)
X[1] = q0_val[1]
Y[1] = q0_val[2]
for i = 2 : n + 1
    si = (i-1) * h
    deltax = 0.
    deltay = 0.
    for k = 1:N
        deltax += func_x(node[k], si)*coe[k]
        deltay += func_y(node[k], si)*coe[k]
    end
    j::Int64 = i
    X[j] = q0_val[1] + deltax
    Y[j] = q0_val[2] + deltay
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
