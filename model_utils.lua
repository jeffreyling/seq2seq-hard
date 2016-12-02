function clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function adagradStep(x, dfdx, eta, state)
   if not state.var then
      state.var  = torch.Tensor():typeAs(x):resizeAs(x):zero()
      state.std = torch.Tensor():typeAs(x):resizeAs(x)
   end

   state.var:addcmul(1, dfdx, dfdx)
   state.std:sqrt(state.var)
   x:addcdiv(-eta, dfdx, state.std:add(1e-10))
end

function get_indices(t, source_char_l)
  local t1 = math.floor((t-1)/source_char_l) + 1
  local t2 = ((t-1) % source_char_l) + 1
  return t1, t2
end
