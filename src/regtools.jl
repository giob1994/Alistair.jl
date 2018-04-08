# -----------------------------------------------
# Mixed Support Functions:
# -----------------------------------------------

macro __test(exp)
    print("[ ", exp, " ]")
    try 
        eval(exp)
        println(" - Test Passed")
    catch
        println(" - Test Failed")
        error("#--- TESTING ABORTED ---#")
    end
end

macro p(exp)
    print("Calling [ ", exp, " ]")
    eval(exp)
end

function present(in::Any)
    show(IOContext(STDOUT, limit=true), "text/plain", in)
end

function isvectorarray{T<:Number}(A::Array{T})
    if !issingletonarray(A)
        return (size(A)[1] == 1 || size(A)[2] == 1)
    end
end

function isrowarray{T<:Number}(A::Array{T})
    if !issingletonarray(A)
        return size(A)[1] == 1
    else
        return false
    end
end

function iscolumnarray{T<:Number}(A::Array{T})
    if !issingletonarray(A)
        return size(A)[2] == 1
    else
        return false
    end
end

function issingletonarray{T<:Number}(A::Array{T})
    return (length(size(A)) == 1 && length(A) == 1)
end

function row{T<:Number}(A::Array{T,1})
    if !issingletonarray(A)
        return size(A)[1] > size(A)[2] ? A' : A
    else
        return A
    end
end

function aprod{T1<:Number, T2<:Number}(A::Array{T1}, B::Array{T2})
    if (issingletonarray(A) || issingletonarray(B))
        return A .* B
    else
        return A * B
    end
end

function aprod{T1<:Number, T2<:Number}(A::T1, B::Array{T2})
    return A .* B
end

function aprod{T1<:Number, T2<:Number}(A::Array{T1}, B::T2)
    return A .* B
end

function hasintercept{T<:Number}(A::Array{T})
    if !issingletonarray(A)
        n = size(A)[1]
        return A[:,1] == ones(n)
    else
        return false
    end
end

function addintercept{T<:Number}(A::Array{T})
    if (!hasintercept(A)) 
        return hcat(ones(size(A)[1],1), A) 
    else
        return A
    end
end

function nointercept{T<:Number}(A::Array{T})
    if (hasintercept(A)) 
        return A[:,2:end]
    else
        return A
    end
end

function iscategorical{T<:Number}(A::Array{T})
    return mapreduce(x -> (x == 1 || x == 0), &, A)
end