# Sonnet

Base by DeepSpeed-0.13.0

# Update recored

## 9.27

update

/csrc/adam/cpu_adam_impl.cpp: update Sonnet Optimizer
/csrc/adam/cpu_adam.cpp: update m.der for Sonnet
/csrc/includes/cpu_adam.h: update Sonnet Optimizer

/op_builder/cpu_adam.py: update Sonnet
/op_builder/cpu/cpu_adam.py: update Sonnet

## 9.29

update

/deepspeed/adam/cpu_adam.py update Sonnet Optimizer

some question:
  param_group need optimize

## 10.4

update 

/deepspeed/runtime/engine.py
/deepspeed/runtime/zero/stage1_and_stage2.py update

optimizer congfigure in engine

```

if has_optimizer:
  self._configure_optimizer(optimizer, model_parameters)
  self._configure_lr_scheduler(lr_scheduler)
  self._report_progress(0)

  _configure_optimizer(optimizer, model_parameters)

```