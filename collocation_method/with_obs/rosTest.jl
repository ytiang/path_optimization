#!/home/yangt/workspace/software/julia/bin/julia
# julia 的变量全是引用！
using RobotOS
@rosimport grid_map_msgs.msg:GridMap
@rosimport geometry_msgs.msg:Point,PoseStamped
@rosimport nav_msgs.msg:Path
rostypegen()
using grid_map_msgs.msg
using geometry_msgs.msg
using nav_msgs.msg
using FastGaussQuadrature
using JuMP
# using Ipopt
using NLopt
using PyPlot
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
# map related
column = []
row = []
stride = 0
distance_map = []
diffx_map = []
diffy_map = []
resolution = 0.1

# init model
k_max = 0.2 # max kppa
s_len_max = 150.0
lb = [-k_max, -k_max, 0]
ub = [k_max, k_max, s_len_max]
# set model
# mdl = Model(solver = IpoptSolver())
mdl = Model(solver = NLoptSolver(algorithm=:LD_SLSQP))
@variable(mdl, p[i=1:3], lowerbound=lb[i], upperbound=ub[i])
initial_guess = [0.039, -0.031, 97.03]
for i=1:3
    setvalue(p[i], initial_guess[i])
end
@NLparameter(mdl, q0[i=1:3] == 0)
@NLparameter(mdl, qf[i=1:3] == 0)
setvalue(q0[1], 0.)
setvalue(q0[2], 0.)
setvalue(q0[3], 0.)
setvalue(qf[1], 40.)
setvalue(qf[2], 40.)
setvalue(qf[3], 0.)
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
function disFunc(x, y)
    i = round(x / resolution)
    j = round(y / resolution)
    index::Int32 = i + j * stride
    dist = distance_map[index]
    value = 0
    if dist <= 0.
        value = dis_threshold - dist
    elseif 0 < dist <= dis_threshold
        value = (dist - dis_threshold)^2 / dis_threshold
    end
    return value
end
function ∇disFunc(x, y, dis)
    i::Int32 = round(x / resolution)
    j::Int32 = round(y / resolution)
    index::Int32 = i + j * stride
    ∇ = [0, 0]
    if dis <= 0.
        ∇[1] = -diffx_map[index]
        ∇[2] = -diffy_map[index]
    elseif 0 < dist <= dis_threshold
        ∇[1] = 2/dis_threshold*(dis-dis_threshold)*diffx_map[index]
        ∇[2] = 2/dis_threshold*(dis-dis_threshold)*diffy_map[index]
    end
    return ∇
end
#objective
function myObject(p1, p2, p3)
    local b = -(11*k0 - 18*p1 + 9*p2)/(2*p3)
    local c = (18*k0 - 45*p1 + 36*p2)/(2*p3^2)
    local d = (-9*k0 + 27*p1 - 27*p2)/(2*p3^3)
    q = getvalue(q0)
    # obstacle term
    J₁ = 0
    for k=1:N
        sₖ = p3/2*(node[k]+1)
        Δxₖ = 0
        Δyₖ = 0
        for j = 1:N
            sₖⱼ = sₖ/2*(node[j]+1)
            θⱼ = q[3] + k0*sₖⱼ + b/2*sₖⱼ^2 + c/3*sₖⱼ^3 + d/4*sₖⱼ^4
            Δxₖ += cos(θⱼ) * coe[j] * sₖ / 2
            Δyₖ += sin(θⱼ) * coe[j] * sₖ / 2
        end
        xₖ = q[1] + Δxₖ
        yₖ = q[2] + Δyₖ
        J₁ += p3/2 * disFunc(xₖ, yₖ) * coe[k] / 2
    end
    # smooth term
    J₂ = (b^2 )/3*(p3^3) +
          (b*c)/2*(p3^4) +
          (2*b*d + c^2)/5*(p3^5) +
          c*d/3*(p3^6) +
         (d^2)/7*(p3^7)
    obj = J₁ + J₂
    return obj
end
function ∇myObject(g, p1, p2, p3)
    local b = -(11*k0 - 18*p1 + 9*p2)/(2*p3)
    local c = (18*k0 - 45*p1 + 36*p2)/(2*p3^2)
    local d = (-9*k0 + 27*p1 - 27*p2)/(2*p3^3)
    g = [0.,0.,0.]
    g1 = [0.,0.,0.]
    g2 = [0.,0.,0.]
    ∂b∂p = [0,0,0]
    ∂c∂p = [0,0,0]
    ∂d∂p = [0,0,0]
    ∂b∂p[1] = 9/p3
    ∂b∂p[2] = -9/(2*p3)
    ∂b∂p[3] =  (11*k0 - 18*p1 + 9*p2)/(2*p3^2)
    ∂c∂p[1] = -45/(2*p3^2)
    ∂c∂p[2] = 18/(p3^2)
    ∂c∂p[3] = -(18*k0 - 45*p1 + 36*p2)/(p3^3)
    ∂d∂p[1] = 27/(2*p3^3)
    ∂d∂p[2] = -27/(2*p3^3)
    ∂d∂p[3] = -2(-9*k0 + 27*p1 - 27*p2)/(p3^4)
    q = getvalue(q0)
    # obstacle term gradient
    for k = 1 : N
        sₖ = p3/2*(node[k]+1)
        θₖ = q[3] + k0*sₖ +  b/2*sₖ^2 + c/3*sₖ^3 + d/4*sₖ^4
        κₖ = k0 + b*sₖ + c*sₖ^2 + d*sₖ^3
        Δxₖ = 0
        Δyₖ = 0
        ∂Xₖ∂p = [0,0,0]
        ∂Yₖ∂p = [0,0,0]
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
        vₖ = disFunc(xₖ, yₖ)
        ∇vₖ = ∇disFunc(xₖ, yₖ, vₖ)
        ∂f∂xₖ = sin(θₖ)^2 * ∇vₖ[1] - cos(θₖ)*sin(θₖ)*∇vₖ[2] + vₖ*κₖ*sin(θₖ)
        ∂f∂yₖ = -cos(θₖ)*sin(θₖ)*∇vₖ[1] + cos(θₖ)^2 *∇vₖ[2] - vₖ*κₖ*cos(θₖ)
        g1[1] += (∂f∂xₖ*∂Xₖ∂p[1] + ∂f∂yₖ*∂Yₖ∂p[1]) * coe[k] * p3/2
        g1[2] += (∂f∂xₖ*∂Xₖ∂p[2] + ∂f∂yₖ*∂Yₖ∂p[3]) * coe[k] * p3/2
        g1[3] += (∂f∂xₖ*∂Xₖ∂p[3] + ∂f∂yₖ*∂Yₖ∂p[3]) * coe[k] * p3/2
    end
    # smooth term gradient
    g2[1] = -3//280 * p3 * (59 * k0 - 72 * p1 + 9 * p2)
    g2[1] = -3//280 * p3 * (74 * k0 + 9 * p1 - 72 * p2)
    g2[3] = 347//420 * k0 ^ 2 - 177//280 * k0 * p1 -
           111//140 * k0 * p2 + 27//70 * p1 ^ 2 -
           27//280 * p1 * p2 + 27//70 * p2 ^ 2
    g[1] = g1[1] + g2[1]
    g[2] = g1[2] + g2[2]
    g[3] = g1[3] + g2[3]
end
JuMP.register(mdl, :myObject, 3, myObject, ∇myObject)
@NLobjective(mdl, Min, myObject(p[1], p[2], p[3]))
# @NLobjective(mdl,
#     Min,
#     (b^2 )/3*(p[3]^3) +
#     (b*c)/2*(p[3]^4) +
#     (2*b*d + c^2)/5*(p[3]^5) +
#     c*d/3*(p[3]^6) +
#     (d^2)/7*(p[3]^7))

function func_x(t, s1)
    b_val = getvalue(b)
    c_val = getvalue(c)
    d_val = getvalue(d)
    q0_val = getvalue(q0)
    s::Float64 =  (s1)/2 + (s1)*t/2
    return cos(q0_val[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4))*s1/2
end
function func_y(t, s1)
    b_val = getvalue(b)
    c_val = getvalue(c)
    d_val = getvalue(d)
    q0_val = getvalue(q0)
    s::Float64 =  (s1)/2 + (s1)*t/2
    return sin(q0_val[3] + k0*s + b_val/2*(s^2) + c_val/3*(s^3) + d_val/4*(s^4))*s1/2
end
# get initial state
# function initialStateCb(msg::Point)
#     setvalue(q0[1], msg.x)
#     setvalue(q0[2], msg.y)
#     setvalue(q0[3], msg.z)
#     global is_get_initial_state = 10
#     # print("received initial state: ", q0, "\n")
# end
# get goal state
function goalStateCb(msg::Point)
    setvalue(qf[1], msg.x)
    setvalue(qf[2], msg.y)
    setvalue(qf[3], msg.z)
    print("x: ", msg.x, " y: ", msg.y, " θ: ", msg.z, "\n")
    global is_get_goal_state = 10
end

# get distance map and gradient map
function mapCb(msg::GridMap)
    global column = msg.data[1].layout.dim[1].size
    global row = msg.data[1].layout.dim[2].size
    global stride = msg.data[1].layout.dim[2].stride
    global distance_map = msg.data[1].data
    global diffx_map = msg.data[2].data
    global diffy_map = msg.data[3].data
    global resolution = msg.info.resolution
    # print("In Cb, row: ",row, " ,column: ", column, " ,stride: ", stride,"\n")
end

function main()
    #init ros related
    init_node("julia_node")
    # initial_state_sub = Subscriber("init_state", Point, initialStateCb, queue_size=10)
    goal_state_sub = Subscriber("goal_state", Point, goalStateCb, queue_size=10)
    distance_map_sub = Subscriber("env_map", GridMap, mapCb, queue_size=10)
    path_pub = Publisher("path", Path, queue_size=10)
    loop_rate = Rate(0.5)
    global is_get_goal_state
    global is_get_initial_state
    global mdl
    print("init_flag: ", is_get_initial_state, ", goal_flag: ", is_get_goal_state, "\n")
    while ! is_shutdown()
        if is_get_goal_state > 0 #&& is_get_initial_state > 0
            @time status = solve(mdl)

            path_msg = Path()
            path_msg.header.frame_id = "odom"
            path_msg.header.stamp = get_rostime()
            q0_val = getvalue(q0)
            p_val = getvalue(p)
            n = round(p_val[3])
            h = p_val[3] / n
            for i = 2 : n + 1
                local temp_pose = PoseStamped()
                temp_pose.header = path_msg.header
                si = (i-1) * h
                deltax = 0.
                deltay = 0.
                for k = 1:N
                    deltax += func_x(node[k], si)*coe[k]
                    deltay += func_y(node[k], si)*coe[k]
                end

                temp_pose.pose.position.x = q0_val[1] + deltax
                temp_pose.pose.position.y = q0_val[2] + deltay
                push!(path_msg.poses, temp_pose)
            end
            publish(path_pub, path_msg)

            is_get_goal_state = 0
        end
        # print("In main, row: ",row, " ,column: ", column, " ,stride: ", stride,"\n")
        rossleep(loop_rate)
        # print(path)
    end
end

if ! isinteractive()
    main()
end
