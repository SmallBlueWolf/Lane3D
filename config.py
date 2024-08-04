img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
input_size = (360, 480)

feat_y_steps = [5,  10,  15,  20,  30,  40,  50,  60,  80,  100]
anchor_y_steps = [5,  10,  15,  20,  30,  40,  50,  60,  80,  100]
anchor_len = len(anchor_y_steps)


# model set
model = dict(
    type = 'Anchor3DLane',
    backbone=dict(
    type='ResNetV1c',
    depth=18,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    dilations=(1, 1, 2, 4),
    strides=(1, 2, 1, 1),
    style='pytorch'),
    pretrained = 'pretrained/resnet18_v1c-b5776b93.pth',
    y_steps = anchor_y_steps,
    feat_y_steps = feat_y_steps,
    anchor_cfg = dict(pitches = [5, 2, 1, 0, -1, -2, -5],
        yaws = [30, 20, 15, 10, 7, 5, 3, 1, 0, -1, -3, -5, -7, -10, -15, -20, -30],
        num_x = 45, distances=[3,]),
    db_cfg = dict(
        org_h = 1080,
        org_w = 1920,
        resize_h = 360,
        resize_w = 480,
        ipm_h = 208,
        ipm_w = 128,
        pitch = 3,
        cam_height = 1.55,
        crop_y = 0,
        K = [[2015., 0., 960.], [0., 2015., 540.], [0., 0., 1.]],
        top_view_region = [[-10, 103], [10, 103], [-10, 3], [10, 3]],
        max_2dpoints = 10,
    ),
    attn_dim = 64,
    iter_reg = 0,
    drop_out=0.,
    num_heads = 2,
    dim_feedforward = 128,
    pre_norm = False,
    feat_size = (45, 60),
    num_category = 2,
    loss_lane = dict(
        type = 'LaneLoss',
        loss_weights = dict(cls_loss = 1,
                            reg_losses_x = 1,
                            reg_losses_z = 1,
                            reg_losses_vis = 1),
        assign_cfg = dict(
            type = 'TopkAssigner',
            pos_k = 3,
            neg_k = 450,
            anchor_len = anchor_len,
            metric = 'Euclidean'
        ),
        anchor_len = anchor_len,
        anchor_steps=anchor_y_steps,
    ),
    train_cfg = dict(
        nms_thres = 0,
        conf_threshold = 0),
    test_cfg = dict(
        nms_thres = 2,
        conf_threshold = 0.5,
        test_conf = 0.7,
        refine_vis = True,
        vis_thresh = 0.5
    )
)