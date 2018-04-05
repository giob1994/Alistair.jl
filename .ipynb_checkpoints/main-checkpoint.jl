struct fitresult
        beta
        residuals
        mse
        converged::Bool

        function fitresult(modelfun, X, Y, optimresult::AbsRes)
            beta = Optim.minimizer(optimresult)
            residuals = (Y - modelfun(X, beta)).^2
            mse = Optim.minimum(optimresult)
            converged = Optim.converged(optimresult)
            new(beta, residuals, mse, converged)
        end
end

function nlinfit(modelfun, X, Y, beta0, method::AbsOptim=NewtonTrustRegion())
        if true # test__modelfun(modelfun, X[1:2,:], beta0)
            
            # Residual function given the model:
            residual(beta_) = sum((Y - modelfun(X, beta_)).^2)
        
            optimresult = optimize(residual, beta0, method)
            result = fitresult(modelfun, X, Y, optimresult)
            
            return result
        
        else
        
            error("Something is wrong!")
        
        end

end

function test__modelfun(modelfun, x, b)
        try
            modelfun(x, b)
            return true
        catch
            error("modelfun is not correctly defined")
        end
end
        