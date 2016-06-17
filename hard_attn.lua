-- From dpnn library
--
require 'nn'

function nn.Module:toBatch(tensor, nDim, batchDim)
  local batchDim = batchDim or 1
  if tensor:dim() == nDim then
    self.dpnn_online = true
    local size = tensor:size():totable()
    table.insert(size, batchDim, 1)
    tensor = tensor:view(table.unpack(size))
  else
    self.dpnn_online = false
  end
  return tensor
end

function nn.Module:reinforce(reward)
  if self.modules then
    for i, module in ipairs(self.modules) do
      module:reinforce(reward)
    end
  end
end

nn.Criterion.toBatch = nn.Module.toBatch

--
------------------------------------------------------------------------
--[[ Reinforce ]]--
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Abstract class for modules that use the REINFORCE algorithm (ref A).
-- The reinforce(reward) method is called by a special Reward Criterion.
-- After which, when backward is called, the reward will be used to 
-- generate gradInputs. The gradOutput is usually ignored.
------------------------------------------------------------------------
local Reinforce, parent = torch.class("nn.Reinforce", "nn.Module")

function Reinforce:__init(stochastic)
   parent.__init(self)
   -- true makes it stochastic during evaluation and training
   -- false makes it stochastic only during training
   self.stochastic = stochastic
end

-- a Reward Criterion will call this
function Reinforce:reinforce(reward)
   --parent.reinforce(self, reward)
   if self.reward == nil then
     self.reward = reward
   else
     self.reward:add(reward)
   end
end

function Reinforce:updateOutput(input)
   self.output:set(input)
end

function Reinforce:updateGradInput(input, gradOutput)
   local reward = self:rewardAs(input)
   self.gradInput:resizeAs(reward):copy(reward)
end

-- this can be called by updateGradInput
function Reinforce:rewardAs(input)
   assert(self.reward:dim() == 1)
   if input:isSameSizeAs(self.reward) then
      return self.reward
   else
      if self.reward:size(1) ~= input:size(1) then
         -- assume input is in online-mode
         input = self:toBatch(input, input:dim())
         assert(self.reward:size(1) == input:size(1), self.reward:size(1).." ~= "..input:size(1))
      end
      self._reward = self._reward or self.reward.new()
      self.__reward = self.__reward or self.reward.new()
      local size = input:size():fill(1):totable()
      size[1] = self.reward:size(1)
      self._reward:view(self.reward, table.unpack(size))
      self.__reward:expandAs(self._reward, input)
      return self.__reward
   end
end

------------------------------------------------------------------------
--[[ ReinforceCategorical ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are a vector of categorical prob : (p[1], p[2], ..., p[k]) 
-- Ouputs are samples drawn from this distribution.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.230-236) which is 
-- implemented through the nn.Module:reinforce(r,b) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
local ReinforceCategorical, parent = torch.class("nn.ReinforceCategorical", "nn.Reinforce")

function ReinforceCategorical:__init(semi_sampling_p, stochastic)
  parent.__init(self, stochastic)
  self.semi_sampling_p = semi_sampling_p or 1
  self.entropy_scale = 0
end

function ReinforceCategorical:updateOutput(input)
   self.output:resizeAs(input)
   self._index = self._index or ((torch.type(input) == 'torch.CudaTensor') and torch.CudaTensor() or torch.LongTensor())
   self._do_sample = (torch.uniform() < self.semi_sampling_p)
   if self._do_sample and (self.stochastic or self.train ~= false) then
      -- sample from categorical with p = input
      self._input = self._input or input.new()
      -- prevent division by zero error (see updateGradInput)
      self._input:resizeAs(input):copy(input):add(0.00000001) 
      input.multinomial(self._index, input, 1)
      -- one hot encoding
      self.output:zero()
      self.output:scatter(2, self._index, 1)
   else
      -- use p for evaluation
      self.output:copy(input)
   end
   return self.output
end

function ReinforceCategorical:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : categorical probability mass function
   -- x : the sampled indices (one per sample) (self.output)
   -- p : probability vector (p[1], p[2], ..., p[k]) 
   -- derivative of log categorical w.r.t. p
   -- d ln(f(x,p))     1/p[i]    if i = x  
   -- ------------ =   
   --     d p          0         otherwise
   self.gradInput:resizeAs(input):zero()
   self.gradInput:copy(self.output)
   self._input = self._input or input.new()
   -- prevent division by zero error
   self._input:resizeAs(input):copy(input):add(0.00000001) 
   self.gradInput:cdiv(self._input)
   
   -- multiply by reward 
   self.gradInput:cmul(self:rewardAs(input))
   -- add entropy term
   self._gradEnt = self._input:clone()
   self._gradEnt:log():add(1)
   self.gradInput:add(self.entropy_scale, self._gradEnt)

   -- multiply by -1 ( gradient descent on input )
   self.gradInput:mul(-1)
   return self.gradInput
end

function ReinforceCategorical:type(type, tc)
   self._index = nil
   return parent.type(self, type, tc)
end


-- Modified ClassNLLCriterion
local ReinforceNLLCriterion, parent = torch.class("nn.ReinforceNLLCriterion", "nn.Criterion")

function ReinforceNLLCriterion:__init(modules, weights, sizeAverage, scale)
   parent.__init(self)
   self.modules = modules -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   --self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
   if weights then
     assert(weights:dim() == 1, "weights input should be 1-D Tensor")
     self.weights = weights
   end

   self.gradInput = torch.Tensor()
end

function ReinforceNLLCriterion:updateOutput(inputTable, target)
   assert(torch.type(inputTable) == 'table')
   local input = inputTable[1]
   local mask = inputTable[3]

   if type(target) == 'number' then
     if input:type() ~= 'torch.CudaTensor' then
       self.target = self.target:long()
     end
     self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
     self.target = target
   else
     self.target = target:long()
   end

   self.reward = self.reward or input.new()
   self.reward = input:gather(2,target:view(target:size(1), 1))
   self.reward:resize(input:size(1))
   self.reward:maskedFill(mask, 0) -- zero out padding samples

   -- loss = -sum(reward) aka NLL
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/input:size(1)
   end
   return self.output
end

-- TODO: consider making the baseline learned
function ReinforceNLLCriterion:updateGradInput(inputTable, target)
  -- t is timestep of LSTM
  local input = inputTable[1]
  local baseline = inputTable[2]
  local mask = inputTable[3]
  local t = inputTable[4]

   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-baseline)
   self.vrReward:mul(self.scale) -- scale 
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   -- broadcast reward to modules
   for i, module in ipairs(self.modules) do
     module:reinforce(self.vrReward)  
     if i > t then break end -- reward only modules from t or before
   end

   -- gradInput
   self.gradInput:resizeAs(input):zero()
   local ones = input.new():resize(target:size(1), 1):fill(1)
   if input:type() == 'torch.CudaTensor' then
     ones = ones:cuda()
   end
   self.gradInput:scatter(2, target:view(target:size(1), 1), -ones)
   self.gradInput:maskedFill(mask:view(mask:size(1),1):expand(self.gradInput:size()), 0) -- zero out padding samples

   if self.sizeAverage then
     self.gradInput:div(input:size(1))
   end
   return self.gradInput
end

function ReinforceNLLCriterion:type(type)
   --self._maxVal = nil
   --self._maxIdx = nil
   --self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
