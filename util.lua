--LinearNoBias from elements library
local LinearNoBias, Linear = torch.class('nn.LinearNoBias', 'nn.Linear')

function LinearNoBias:__init(inputSize, outputSize)
   nn.Module.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)

   self:reset()
end

function LinearNoBias:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end

   return self
end

function LinearNoBias:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      self.output:mv(self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearNoBias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   end
end


local ViewAs = torch.class('nn.ViewAs', 'nn.Module')
-- Views input[1] based on first ndim sizes of input[2]

function ViewAs:__init(ndim)
  nn.Module.__init(self)
  self.ndim = ndim
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function ViewAs:updateOutput(input)
  self.output = self.output or input.new()

  assert(#input == 2, 'ViewAs can only take 2 inputs')
  if self.ndim then
    local sizes = {}
    for i = 1, self.ndim do
      sizes[#sizes+1] = input[2]:size(i)
    end
    self.output:view(input[1], table.unpack(sizes))
  else
    local sizes = input[2]:size()
    self.output:view(input[1], sizes)
  end
  return self.output
end

function ViewAs:updateGradInput(input, gradOutput)
  self.gradInput[2]:resizeAs(input[2]):zero() -- unused

  self.gradInput[1] = self.gradInput[1] or gradOutput.new()
  self.gradInput[1]:view(gradOutput, input[1]:size())
  return self.gradInput
end



local ReplicateAs = torch.class('nn.ReplicateAs', 'nn.Module')
-- Replicates dim m of input[1] based on dim n of input[2]
-- basically copies Replicate

function ReplicateAs:__init(in_dim, template_dim)
  nn.Module.__init(self)
  self.in_dim = in_dim
  self.template_dim = template_dim
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function ReplicateAs:updateOutput(input)
  assert(#input == 2, 'needs 2 inputs')
  local rdim = self.in_dim
  local ntimes = input[2]:size(self.template_dim)
  input = input[1]
  local sz = torch.LongStorage(input:dim() + 1)
  sz[rdim] = ntimes
  for i = 1,input:dim() do
    local offset = 0
    if i >= rdim then offset = 1 end
    sz[i+offset] = input:size(i)
  end
  local st = torch.LongStorage(input:dim() + 1)
  st[rdim] = 0
  for i = 1,input:dim() do
    local offset = 0
    if i >= rdim then offset = 1 end
    st[i+offset] = input:stride(i)
  end
  self.output:set(input:storage(), input:storageOffset(), sz, st)
  return self.output
end

function ReplicateAs:updateGradInput(input, gradOutput)
  self.gradInput[2]:resizeAs(input[2]):zero() -- unused

  input = input[1]
  self.gradInput[1]:resizeAs(input):zero()
  local rdim = self.in_dim
  local sz = torch.LongStorage(input:dim() + 1)
  sz[rdim] = 1
  for i = 1, input:dim() do
    local offset = 0
    if i >= rdim then offset = 1 end
    sz[i+offset] = input:size(i)
  end
  local gradInput = self.gradInput[1]:view(sz)
  gradInput:sum(gradOutput, rdim)

  return self.gradInput
end
