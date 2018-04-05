abstract type RegressionType end

struct OLS
    robust::Bool

    function OLS(robust=true)
        new(robust)
    end
end 

