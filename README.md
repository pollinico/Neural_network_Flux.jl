# Neural network in Julia with Flux.jl

Function approximated:
```julia
function myFun(x1, x2)
    f = x1 * exp(-x1^2-x2^2)
    return f
end
```

<img src="loss.png" alt="loss" width="500"/>   

<img src="function_approximation.png" alt="function approximation" width="500"/> 
