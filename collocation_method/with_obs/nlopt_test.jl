#!/home/yangt/workspace/software/julia/bin/julia
# julia 的变量全是引用！
# 出现InExactError的原因可能有：使用变量未声明、声明时默认为整型，赋值时却为float型，超出了范围，比如0和0.的区别
using FastGaussQuadrature
using JuMP
using Ipopt
# using NLopt
using PyPlot
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
q0 = (5, 5, 0) #initial configuration
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
JuMP.register(mdl, :dist, 2, dist, ∇dist)
for i = 1:num
    @NLconstraint(mdl, dist(x[i], y[i]) <= 10.0)
end
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
