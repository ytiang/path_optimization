#!/home/yangt/workspace/software/julia/bin/julia
# julia 的变量全是引用！
# 出现InExactError的原因可能有：使用变量未声明、声明时默认为整型，赋值时却为float型，超出了范围，比如0和0.的区别
# using RobotOS
# @rosimport geometry_msgs.msg:Point,PoseStamped
# @rosimport nav_msgs.msg:Path
# rostypegen()
using FastGaussQuadrature
using JuMP
using Ipopt
# using NLopt
using PyPlot
# using Plotly
using CSV
# using geometry_msgs.msg
# using nav_msgs.msg

FILE1 = "/home/yangt/catkin_ws/src/path_optimization/src/julia/distance.csv"
FILE2 = "/home/yangt/catkin_ws/src/path_optimization/src/julia/gradient.csv"
FILE3 = "/home/yangt/catkin_ws/src/path_optimization/src/julia/obstacle.csv"

dist_file = CSV.read(FILE1; header = true)
grad_file = CSV.read(FILE2; header = true)
obtacle_file = CSV.read(FILE3; header = true)
# map parameters:
stride = 800
column = 800
row = 800
obs_x = Array(obtacle_file[:x])
obs_y = Array(obtacle_file[:y])
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
k_max = 0.2# max kppa
s_len_max = 500.0
lb = [-k_max, -k_max, -k_max, -k_max, 0]        #lower bound
ub = [k_max, k_max, k_max, k_max, s_len_max]    # upper bound
# set model
# initial_guess = [0.0623997, -0.02136, 45.4845]
initial_guess = [0.0, 0.0, 0.0, 0.0, 45.4845]
initial_state = [5., 5., 0.]
goal_state = [40.0, 40.0, 0.0]


function mapAt(x, y, map)
    indexx = max(min(ceil(x / resolution),800),0)
    indexy = max(min(ceil(y / resolution),800),1)
    dist = 0.
    # print("x, y: ", x," ",y," indx, indy: ",indexx," ",indexy,"\n")
    # if indexx > row || indexy > column
    #     print("exceed map litmit!\n")
    #     return -10
    # end

    i::Int32 = row - indexx + 1
    j::Int32 = column - indexy
    index::Int32 = min(i+j*stride, 640000)
    if index > 640000
        print("***************get: ", index, ", ",i ,", ", j,"; ","input: ",x,", ",y,"\n")
    end
    dist = map[index]
    return dist
end

function pathPt(s, p1, p2, p3, p4, p5, q)
    Δx = 0.
    Δy = 0.
    pt = [0., 0.]
    # local q = getvalue(q0)
    local b_val = (-137k0+300p1-300p2+200p3-75p4)/(12p5)
    local c_val = 25(45k0-154p1+214p2-156p3+61p4)/(24p5^2)
    local d_val = -125(17k0-71p1+118p2-98p3+41p4)/(24p5^3)
    local e_val = 625(3k0-14p1+26p2-24p3+11p4)/(24p5^4)
    local f_val = -625(k0-5p1+10p2-10p3+5p4)/(24p5^5)
    for j = 1:N
        sⱼ = s/2*(node[j]+1)
        θⱼ = q[3] + k0*sⱼ + b_val/2*sⱼ^2 + c_val/3*sⱼ^3 + d_val/4*sⱼ^4 + e_val/5*sⱼ^5 + f_val/6*sⱼ^6
        Δx += cos(θⱼ) * coe[j] * s / 2
        Δy += sin(θⱼ) * coe[j] * s / 2
    end
    pt[1] = q[1] + Δx
    pt[2] = q[2] + Δy
    return pt
end
#objective
function myObject(p1, p2, p3, p4, p5)
    local b = (-137k0+300p1-300p2+200p3-75p4)/(12p5)
    local c = 25(45k0-154p1+214p2-156p3+61p4)/(24p5^2)
    local d = -125(17k0-71p1+118p2-98p3+41p4)/(24p5^3)
    local e = 625(3k0-14p1+26p2-24p3+11p4)/(24p5^4)
    local f = -625(k0-5p1+10p2-10p3+5p4)/(24p5^5)
    q = getvalue(q0)
    # obstacle term
    J₁ = 0.
    for k=1:N
        sₖ = p5/2*(node[k]+1)
        rₖ = pathPt(sₖ, p1, p2, p3, p4, p5, q)
        # print("yyyyyyyyyyyyyyyyyyyyy: ",J₁,"\n")
        J₁ += mapAt(rₖ[1], rₖ[2], distance_map) * coe[k] * p5/2
    end
    # smooth term
    J₂ = 0.
    J₂ = k0^2*p5 + k0*b*p5^2 +
        (b^2+2k0*c)/3*p5^3 +
        (2b*c+2k0*d)/4*p5^4 +
        (2b*d+c^2+2k0*e)/5*p5^5 +
        (2b*e+2c*d+2k0*f)/6*p5^6 +
        (2b*f+2c*e+d^2)/7*p5^7 +
        (2c*f+2d*e)/8*p5^8 +
        (2d*f+e^2)/9*p5^9 +
        (e*f)/5*p5^10 +
        f^2/11*p5^11
    obj = 100*J₁ + J₂
    # obj = J₂
    print("************J1, J2, obj: ",J₁," ",J₂," ",obj,"\n")
    return obj
end
function ∇myObject(g, p1, p2, p3, p4, p5)
    local b = (-137k0+300p1-300p2+200p3-75p4)/(12p5)
    local c = 25(45k0-154p1+214p2-156p3+61p4)/(24p5^2)
    local d = -125(17k0-71p1+118p2-98p3+41p4)/(24p5^3)
    local e = 625(3k0-14p1+26p2-24p3+11p4)/(24p5^4)
    local f = -625(k0-5p1+10p2-10p3+5p4)/(24p5^5)
    g1 = [0., 0., 0., 0., 0.]
    g2 = [0., 0., 0., 0., 0.]
    g = [0., 0., 0., 0., 0.]
    ∂b∂p = [0., 0., 0., 0., 0.]
    ∂c∂p = [0., 0., 0., 0., 0.]
    ∂d∂p = [0., 0., 0., 0., 0.]
    ∂e∂p = [0., 0., 0., 0., 0.]
    ∂f∂p = [0., 0., 0., 0., 0.]
    ∂b∂p[1] = 300/(12*p5)
    ∂b∂p[2] = -300/(12*p5)
    ∂b∂p[3] =  200/(12*p5)
    ∂b∂p[4] = -75/(12*p5)
    ∂b∂p[5] =  -(-137k0+300p1-300p2+200p3-75p4)/(12*p5^2)
    ∂c∂p[1] = -25*154/(24p5^2)
    ∂c∂p[2] = 25*214/(24p5^2)
    ∂c∂p[3] = -25*156/(24p5^2)
    ∂c∂p[4] = 25*61/(24p5^2)
    ∂c∂p[5] = -25(45k0-154p1+214p2-156p3+61p4)/(12p5^3)
    ∂d∂p[1] = 125*71/(24p5^3)
    ∂d∂p[2] = -125*118/(24p5^3)
    ∂d∂p[3] = 125*98/(24p5^3)
    ∂d∂p[4] = -125*41/(24p5^3)
    ∂d∂p[5] = 125(17k0-71p1+118p2-98p3+41p4)/(8p5^4)
    ∂e∂p[1] = -625*14/(24p5^4)
    ∂e∂p[2] = 625*26/(24p5^4)
    ∂e∂p[3] = -625*24/(24p5^4)
    ∂e∂p[4] = 625*11/(24p5^4)
    ∂e∂p[5] = -625(3k0-14p1+26p2-24p3+11p4)/(6p5^5)
    ∂f∂p[1] = 625*5/(24p5^5)
    ∂f∂p[2] = -625*10/(24p5^5)
    ∂f∂p[3] = 625*10/(24p5^5)
    ∂f∂p[4] = -625*5/(24p5^5)
    ∂f∂p[5] = 5*625(k0-5p1+10p2-10p3+5p4)/(24p5^6)
    q = getvalue(q0)
    # # # obstacle term gradient
    for k = 1 : N
        sₖ = p5/2*(node[k]+1)
        θₖ = q[3] + k0*sₖ +  b/2*sₖ^2 + c/3*sₖ^3 + d/4*sₖ^4 + e/5*sₖ^5 + f/6*sₖ^6
        κₖ = k0 + b*sₖ + c*sₖ^2 + d*sₖ^3 + e*sₖ^4 + f*sₖ^4
        Δxₖ = 0.
        Δyₖ = 0.
        ∂Xₖ∂p = [0., 0., 0., 0., 0.]
        ∂Yₖ∂p = [0., 0., 0., 0., 0.]
        for j = 1:N
            sₖⱼ = sₖ/2*(node[j]+1)
            θⱼ = q[3] + k0*sₖⱼ + b/2*sₖⱼ^2 + c/3*sₖⱼ^3 + d/4*sₖⱼ^4 + e/5*sₖⱼ^5 + f/6*sₖⱼ^6
            Δxₖ += cos(θⱼ) * coe[j] * sₖ / 2
            Δyₖ += sin(θⱼ) * coe[j] * sₖ / 2
            for i=1:5
                ∂Xₖ∂p[i] += -sin(θⱼ)* (1/2*∂b∂p[i]*sₖⱼ^2 + 1/3*∂c∂p[i]*sₖⱼ^3 + 1/4*∂d∂p[i]*sₖⱼ^4 + 1/5*∂e∂p[i]*sₖⱼ^5 + 1/6*∂f∂p[i]*sₖⱼ^6) * coe[j]*sₖ / 2
                ∂Yₖ∂p[i] += cos(θⱼ)* (1/2*∂b∂p[i]*sₖⱼ^2 + 1/3*∂c∂p[i]*sₖⱼ^3 + 1/4*∂d∂p[i]*sₖⱼ^4 + 1/5*∂e∂p[i]*sₖⱼ^5 + 1/6*∂f∂p[i]*sₖⱼ^6) * coe[j]*sₖ / 2
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
        # ∇vₖ[1] = ∇vₖ[1] / normv
        # ∇vₖ[2] = ∇vₖ[2] / normv

        # ∂f∂xₖ = -sin(θₖ)^2 * ∇vₖ[1] - cos(θₖ)*sin(θₖ)*∇vₖ[2] + vₖ*κₖ*sin(θₖ)
        # ∂f∂yₖ = -cos(θₖ)*sin(θₖ)*∇vₖ[1] + cos(θₖ)^2 *∇vₖ[2] - vₖ*κₖ*cos(θₖ)
        # print("∇dis, ∂f∂xₖ, ∂f∂yₖ: ", ∇vₖ ," ,", ∂f∂xₖ, " ,", ∂f∂yₖ,"\n")
        # ∂f∂xₖ = ∇vₖ[1]
        # ∂f∂yₖ = ∇vₖ[2]
        for i=1:5
            g1[i] += (∂f∂xₖ*∂Xₖ∂p[i] + ∂f∂yₖ*∂Yₖ∂p[i]) * coe[k] * p5/2
        end
    end
    print("************g1: ", g1[1], " ,", g1[2], " ,", g1[3], " ,", g1[4], "\n")
    # smooth term gradient
    g2[1] = ((-89663475 * k0 + 448575500 * p1 - 897484250 * p2 + 897634500 * p3 - 452958875 * p4) * p5 ^ 2 + (197072500 * k0 - 985468750 * p1 + 1971562500 * p2 - 1971250000 * p3 + 990625000 * p4) * p5 - 109375000 * k0 + 546875000 * p1 - 1093750000 * p2 + 1093750000 * p3 - 546875000 * p4) / p5 / 145152
    g2[2] = ((179436900 * k0 - 897484250 * p1 + 1795943000 * p2 - 1796194500 * p3 + 906384500 * p4) * p5 ^ 2 - 197218750 * (k0 - 5 * p1 + 10 * p2 - 10 * p3 + 5 * p4) * p5 - 197051250 * (796625//157641 * p4 + k0 - 525750//52547 * p3 + 1577750//157641 * p2 - 46375//9273 * p1) * p5 + 218750000 * k0 - 1093750000 * p1 + 2187500000 * p2 - 2187500000 * p3 + 1093750000 * p4) / p5 / 145152
    g2[3] =((-179472750 * k0 + 897634500 * p1 - 1796194500 * p2 + 1796568000 * p3 - 906546750 * p4) * p5 ^ 2 + 197156250 * (k0 - 5 * p1 + 10 * p2 - 10 * p3 + 5 * p4) * p5 + 197051250 * (796625//157641 * p4 + k0 - 525750//52547 * p3 + 1577750//157641 * p2 - 46375//9273 * p1) * p5 - 218750000 * k0 + 1093750000 * p1 - 2187500000 * p2 + 2187500000 * p3 - 1093750000 * p4) / p5 / 145152

    g2[4] =((90565950 * k0 - 452958875 * p1 + 906384500 * p2 - 906546750 * p3 + 457481750 * p4) * p5 ^ 2 - 99578125 * (k0 - 5 * p1 + 10 * p2 - 10 * p3 + 5 * p4) * p5 - 98525625 * (796625//157641 * p4 + k0 - 525750//52547 * p3 + 1577750//157641 * p2 - 46375//9273 * p1) * p5 + 109375000 * k0 - 546875000 * p1 + 1093750000 * p2 - 1093750000 * p3 + 546875000 * p4) / p5 / 145152

    g2[5] = (2 * (8971059 * k0 ^ 2 + (-89663475 * p1 + 179436900 * p2 - 179472750 * p3 + 90565950 * p4) * k0 + 224287750 * p1 ^ 2 + (-897484250 * p2 + 897634500 * p3 - 452958875 * p4) * p1 + 897971500 * p2 ^ 2 + (-1796194500 * p3 + 906384500 * p4) * p2 + 898284000 * p3 ^ 2 - 906546750 * p3 * p4 + 228740875 * p4 ^ 2) * p5 - 19705125 * (796625//157641 * p4 + k0 - 525750//52547 * p3 + 1577750//157641 * p2 - 46375//9273 * p1) * (k0 - 5 * p1 + 10 * p2 - 10 * p3 + 5 * p4)) / p5 / 145152 - ((8971059 * k0 ^ 2 + (-89663475 * p1 + 179436900 * p2 - 179472750 * p3 + 90565950 * p4) * k0 + 224287750 * p1 ^ 2 + (-897484250 * p2 + 897634500 * p3 - 452958875 * p4) * p1 + 897971500 * p2 ^ 2 + (-1796194500 * p3 + 906384500 * p4) * p2 + 898284000 * p3 ^ 2 - 906546750 * p3 * p4 + 228740875 * p4 ^ 2) * p5 ^ 2 - 19705125 * (796625//157641 * p4 + k0 - 525750//52547 * p3 + 1577750//157641 * p2 - 46375//9273 * p1) * (k0 - 5 * p1 + 10 * p2 - 10 * p3 + 5 * p4) * p5 + 10937500 * (k0 - 5 * p1 + 10 * p2 - 10 * p3 + 5 * p4) ^ 2) / p5 ^ 2 / 145152

    # print("g2: ",g2[1], " ,",g2[2]," ,",g2[3],"\n")
    for i=1:5
        g[i] = 100*g1[i] + g2[i]
        # g[i] = g2[i]
    end
    # print("∇J: ",g[1], " ,",g[2]," ,",g[3],"\n")
end
mdl = Model(solver = IpoptSolver())
# mdl = Model(solver = NLoptSolver(algorithm=:LD_SLSQP))
@variable(mdl, p[i=1:5], lowerbound=lb[i], upperbound=ub[i]) # set variable
@NLparameter(mdl, q0[i=1:3] == initial_state[i])
@NLparameter(mdl, qf[i=1:3] == goal_state[i])
for i=1:5
    setvalue(p[i], initial_guess[i])
end
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

# JuMP.register(mdl, :myObject, 5, myObject, autodiff = true)
JuMP.register(mdl, :myObject, 5, myObject, ∇myObject)
@NLobjective(mdl, Min, myObject(p[1], p[2], p[3], p[4], p[5]))
# @NLobjective(mdl, Min, 1)
# @NLobjective(mdl,
#     Min,
#     k0^2*sf + k0*b*sf^2 +
#     (b^2+2k0*c)/3*sf^3 +
#     (2b*c+2k0*d)/4*sf^4 +
#     (2b*d+c^2+2k0*e)/5*sf^5 +
#     (2b*e+2c*d+2k0*f)/6*sf^6 +
#     (2b*f+2c*e+d^2)/7*sf^7 +
#     (2c*f+2d*e)/8*sf^8 +
#     (2d*f+e^2)/9*sf^9 +
#     (e*f)/5*sf^10 +
#     f^2/11*sf^11)

@time status = solve(mdl)

p_val = getvalue(p)
b_val = getvalue(b)
c_val = getvalue(c)
d_val = getvalue(d)
e_val = getvalue(e)
f_val = getvalue(f)
q0_val = getvalue(q0)
println("p: ", p_val,
        "\nb: ", b_val,
        "\nc: ", c_val,
        "\nd: ", d_val,
        "\ne: ", e_val,
        "\nf: ", f_val)

# plot
function func_x(t, s1)
    s::Float64 =  (s1)/2 + (s1)*t/2
    return cos(q0_val[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4) + e_val/5*(s^5) + f_val/6*(s^6))*s1/2
end
function func_y(t, s1)
    s::Float64 =  (s1)/2 + (s1)*t/2
    return sin(q0_val[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4) + e_val/5*(s^5) + f_val/6*(s^6))*s1/2
end

n = ceil(p_val[5])
h = p_val[5]/n
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
tmp_x = [23.85,23.85, 38.95, 38.95]
tmp_y = [27.45,11.65,11.65,27.45]
rx[:plot](tmp_x, tmp_y)
# for i =1:length(obs_x)
#     PyPlot.scatter(obs_x[i], obs_y[i],alpha=0.1)
# end
# grid("on")


s_val = 0:h:p_val[5]
ks = b_val*s_val + c_val*s_val.^2 + d_val*s_val.^3 + e_val*s_val.^4 + f_val*s_val.^5

kfig = figure("kppa")
kx = kfig[:add_subplot](1,1,1)
kx[:plot](s_val, ks)

tha = b_val/2*(s_val.^2) + c_val/3*(s_val.^3) + d_val/4*(s_val.^4) + e_val/5*s_val.^5 + f_val/6*s_val.^6

tfig = figure("theta")
tx = tfig[:add_subplot](1,1,1)
tx[:plot](s_val, tha)


# init_node("julia_node")
# path_pub = Publisher("path", Path, queue_size=10)
# path_msg = Path()
# path_msg.header.frame_id = "odom"
# path_msg.header.stamp = get_rostime()
# q0_val = getvalue(q0)
# p_val = getvalue(p)
# n = round(p_val[3])
# h = p_val[3] / n
# for i = 2 : n + 1
#     local temp_pose = PoseStamped()
#     temp_pose.header = path_msg.header
#     si = (i-1) * h
#     deltax = 0.
#     deltay = 0.
#     for k = 1:N
#         deltax += func_x(node[k], si)*coe[k]
#         deltay += func_y(node[k], si)*coe[k]
#     end
#
#     temp_pose.pose.position.x = q0_val[1] + deltax
#     temp_pose.pose.position.y = q0_val[2] + deltay
#     push!(path_msg.poses, temp_pose)
# end
# publish(path_pub, path_msg)
