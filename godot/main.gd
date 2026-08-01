extends Node3D

const API_BASE:String = "http://127.0.0.1:8765"
const TEST_SERVER:String = API_BASE + "/infer_test"
const CHAT_SERVER:String = API_BASE + "/chat"
const TEST_WAV:String = "res://test.wav"
const CAMERA_DISTANCE:float = 2.15
const CAMERA_HEIGHT:float = 1.18
const CAMERA_MINIMIZED_DISTANCE:float = 2.65
const AUTO_ROTATE_SPEED:float = 0.34
const LOWER_BODY_BONES:Dictionary = {
	"hips":true,"leftUpperLeg":true,"leftLowerLeg":true,"leftFoot":true,"leftToes":true,
	"rightUpperLeg":true,"rightLowerLeg":true,"rightFoot":true,"rightToes":true}
const TORSO_BONES:Dictionary = {"spine":true,"chest":true,"upperChest":true}
var VRM_BONES:Dictionary = {
	"hips":"J_Bip_C_Hips","spine":"J_Bip_C_Spine","chest":"J_Bip_C_Chest",
	"upperChest":"J_Bip_C_UpperChest","neck":"J_Bip_C_Neck","head":"J_Bip_C_Head",
	"leftShoulder":"J_Bip_L_Shoulder","leftUpperArm":"J_Bip_L_UpperArm",
	"leftLowerArm":"J_Bip_L_LowerArm","leftHand":"J_Bip_L_Hand",
	"rightShoulder":"J_Bip_R_Shoulder","rightUpperArm":"J_Bip_R_UpperArm",
	"rightLowerArm":"J_Bip_R_LowerArm","rightHand":"J_Bip_R_Hand",
	"leftUpperLeg":"J_Bip_L_UpperLeg","leftLowerLeg":"J_Bip_L_LowerLeg",
	"leftFoot":"J_Bip_L_Foot","leftToes":"J_Bip_L_ToeBase",
	"rightUpperLeg":"J_Bip_R_UpperLeg","rightLowerLeg":"J_Bip_R_LowerLeg",
	"rightFoot":"J_Bip_R_Foot","rightToes":"J_Bip_R_ToeBase",
	"leftEye":"J_Adj_L_FaceEye","rightEye":"J_Adj_R_FaceEye"}
const FINGERS:Dictionary = {"Index":"Index","Middle":"Middle","Ring":"Ring","Little":"Little","Thumb":"Thumb"}
const SEGMENTS:Dictionary = {"Proximal":"1","Intermediate":"2","Distal":"3"}

var skeleton: Skeleton3D
var frames: Array = []
var fps:float = 30.0
var playing:bool = false
var bone_indices:Dictionary = {}
var global_rest:Dictionary = {}
var audio_bytes:PackedByteArray = PackedByteArray()
var request_uses_generated_audio:bool = false
var speech_plan:Dictionary = {"emotion":"neutral","gesture_intensity":1.0,"eye_contact":0.7,"speed":1.0}
var emotion_state:String = "neutral"
var gesture_weight:float = 1.0
var eye_contact:float = 0.7
var face_energy:float = 0.0
var last_frame_index:int = -1
var jaw_bone:int = -1
var head_bone:int = -1
var eye_bones:Array = []
var finger_bones:Array = []
var animated_bones:Array[int] = []
var auto_rotate:bool = true
var avatar_yaw:float = PI
var ui_minimized:bool = false
@onready var status: Label = $UI/Panel/VBox/Status
@onready var player: AudioStreamPlayer = $Audio
@onready var request: HTTPRequest = $Request
@onready var chat_input: LineEdit = $UI/Panel/VBox/ChatInput
@onready var reply_label: Label = $UI/Panel/VBox/Reply
@onready var panel: PanelContainer = $UI/Panel
@onready var camera: Camera3D = $Camera3D

var rotate_button:Button
var fullscreen_button:Button
var minimize_button:Button
var controls:HBoxContainer

var request_in_progress:bool = false

func _ready() -> void:
	_configure_mobile_viewport()
	_build_viewport_controls()
	var avatar:Node = load("res://avatar.glb").instantiate()
	$Avatar.add_child(avatar)
	$Avatar.rotation.y = avatar_yaw
	skeleton = _find_skeleton(avatar)
	if skeleton == null:
		status.text = "Avatar has no Skeleton3D"
		return
	_add_finger_names()
	for humanoid in VRM_BONES:
		var index:int = skeleton.find_bone(VRM_BONES[humanoid])
		if index >= 0:
			bone_indices[humanoid] = index
			global_rest[humanoid] = skeleton.get_bone_global_rest(index).basis
			if not animated_bones.has(index):
				animated_bones.append(index)
	head_bone = bone_indices.get("head", -1)
	jaw_bone = skeleton.find_bone("J_Bip_C_Jaw")
	if jaw_bone >= 0 and not animated_bones.has(jaw_bone):
		animated_bones.append(jaw_bone)
	eye_bones = [bone_indices.get("leftEye", -1), bone_indices.get("rightEye", -1)]
	for humanoid in bone_indices:
		if humanoid.contains("Thumb") or humanoid.contains("Index") or humanoid.contains("Middle") or humanoid.contains("Ring") or humanoid.contains("Little"):
			finger_bones.append(bone_indices[humanoid])
	reply_label.text = ""
	_update_camera()
	status.text = "Ready: %d bones. Server %s. Chat, play cached motion, rotate, fullscreen, or minimize." % [bone_indices.size(), API_BASE]
	call_deferred("_on_test_pressed")

func _configure_mobile_viewport() -> void:
	get_tree().root.content_scale_mode = Window.CONTENT_SCALE_MODE_CANVAS_ITEMS
	get_tree().root.content_scale_aspect = Window.CONTENT_SCALE_ASPECT_EXPAND
	if OS.get_name() == "Android":
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
	panel.anchor_left = 0.0
	panel.anchor_top = 0.0
	panel.anchor_right = 0.0
	panel.anchor_bottom = 0.0
	panel.offset_left = 18.0
	panel.offset_top = 18.0
	panel.offset_right = 650.0
	panel.offset_bottom = 270.0

func _build_viewport_controls() -> void:
	controls = HBoxContainer.new()
	controls.name = "ViewportControls"
	controls.anchor_left = 1.0
	controls.anchor_top = 0.0
	controls.anchor_right = 1.0
	controls.anchor_bottom = 0.0
	controls.offset_left = -286.0
	controls.offset_top = 18.0
	controls.offset_right = -18.0
	controls.offset_bottom = 58.0
	controls.alignment = BoxContainer.ALIGNMENT_END
	$UI.add_child(controls)
	rotate_button = _make_control_button("Rotate")
	fullscreen_button = _make_control_button("Full")
	minimize_button = _make_control_button("Min")
	controls.add_child(rotate_button)
	controls.add_child(fullscreen_button)
	controls.add_child(minimize_button)
	rotate_button.pressed.connect(_on_rotate_pressed)
	fullscreen_button.pressed.connect(_on_fullscreen_pressed)
	minimize_button.pressed.connect(_on_minimize_pressed)

func _make_control_button(label:String) -> Button:
	var button:Button = Button.new()
	button.text = label
	button.custom_minimum_size = Vector2(82.0, 38.0)
	button.focus_mode = Control.FOCUS_NONE
	return button

func _set_ui_enabled(enabled:bool)->void:
	var vbox:VBoxContainer = $UI/Panel/VBox
	if vbox:
		for child in vbox.get_children():
			if child is Button or child is LineEdit:
				child.disabled = not enabled

func _add_finger_names() -> void:
	for side in ["left","right"]:
		var prefix:String = "J_Bip_L_" if side == "left" else "J_Bip_R_"
		for finger in FINGERS:
			for segment in SEGMENTS:
				VRM_BONES[side+finger+segment] = prefix+FINGERS[finger]+SEGMENTS[segment]

func _find_skeleton(node: Node) -> Skeleton3D:
	if node is Skeleton3D: return node
	for child in node.get_children():
		var found:Skeleton3D = _find_skeleton(child)
		if found: return found
	return null

func _on_test_pressed() -> void:
	if request_in_progress:
		status.text = "Please wait, request in progress..."
		return
	request_in_progress = true
	_set_ui_enabled(false)
	request_uses_generated_audio = false
	player.stop()
	playing = false
	frames = []
	status.text = "Requesting cached test gesture..."
	var error:Error = request.request(TEST_SERVER,[],HTTPClient.METHOD_POST,"")
	if error != OK:
		request_in_progress = false
		_set_ui_enabled(true)
		status.text = "HTTP request error %s" % error

func _on_speak_pressed() -> void:
	_send_chat_message(chat_input.text)

func _on_chat_input_text_submitted(message:String) -> void:
	_send_chat_message(message)

func _send_chat_message(raw_message:String) -> void:
	if request_in_progress:
		status.text = "Please wait, request in progress..."
		return
	var message:String = chat_input.text.strip_edges()
	if not raw_message.strip_edges().is_empty():
		message = raw_message.strip_edges()
	if message.is_empty():
		status.text = "Enter a message first"
		return
	request_in_progress = true
	_set_ui_enabled(false)
	request_uses_generated_audio = true
	player.stop()
	playing = false
	frames = []
	reply_label.text = ""
	status.text = "Qwen reply, TTS audio, and GestureLSM motion running..."
	var headers:PackedStringArray = PackedStringArray(["Content-Type: application/json"])
	var body:String = JSON.stringify({"message": message})
	var error:Error = request.request(CHAT_SERVER,headers,HTTPClient.METHOD_POST,body)
	if error != OK:
		request_in_progress = false
		_set_ui_enabled(true)
		status.text = "HTTP request error %s" % error

func _on_rotate_pressed() -> void:
	auto_rotate = not auto_rotate
	rotate_button.text = "Rotate" if auto_rotate else "Still"
	status.text = "Auto rotation %s" % ("on" if auto_rotate else "off")

func _on_fullscreen_pressed() -> void:
	var mode:int = DisplayServer.window_get_mode()
	if mode == DisplayServer.WINDOW_MODE_FULLSCREEN or mode == DisplayServer.WINDOW_MODE_EXCLUSIVE_FULLSCREEN:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
		fullscreen_button.text = "Full"
	else:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
		fullscreen_button.text = "Exit"

func _on_minimize_pressed() -> void:
	ui_minimized = not ui_minimized
	panel.visible = not ui_minimized
	minimize_button.text = "Panel" if ui_minimized else "Min"
	_update_camera()

func _notification(what:int) -> void:
	if what == NOTIFICATION_WM_SIZE_CHANGED:
		_update_responsive_ui()
		_update_camera()

func _update_responsive_ui() -> void:
	var size:Vector2 = get_viewport().get_visible_rect().size
	if panel == null:
		return
	var panel_width:float = min(632.0, max(300.0, size.x - 36.0))
	panel.offset_right = panel.offset_left + panel_width
	if controls != null:
		controls.offset_left = -min(286.0, max(220.0, size.x - 36.0))

func _update_camera() -> void:
	if camera == null:
		return
	var size:Vector2 = get_viewport().get_visible_rect().size
	var portrait:bool = size.y > size.x
	var distance:float = CAMERA_MINIMIZED_DISTANCE if ui_minimized else CAMERA_DISTANCE
	var height:float = CAMERA_HEIGHT
	if portrait:
		distance += 0.3
		height += 0.08
	camera.position = Vector3(0.0, height, distance)
	camera.look_at(Vector3(0.0, 1.16, 0.0), Vector3.UP)
	camera.fov = 34.0 if portrait else 30.0

func _on_request_completed(_result:int,code:int,_headers:PackedStringArray,body:PackedByteArray)->void:
	request_in_progress = false
	_set_ui_enabled(true)
	if code != 200:
		status.text = "Inference failed (%d): %s" % [code,body.get_string_from_utf8()]
		return
	var payload:Variant = JSON.parse_string(body.get_string_from_utf8())
	if typeof(payload) != TYPE_DICTIONARY:
		status.text = "Bad response: %s" % payload
		return
	var payload_dict:Dictionary = payload as Dictionary
	if payload_dict.has("error"):
		status.text = "Bad response: %s" % payload_dict
		return
	frames = payload_dict.get("frames", []) as Array
	fps = float(payload_dict.get("fps", 30.0))
	if payload_dict.has("speech_plan"):
		speech_plan = payload_dict.get("speech_plan", speech_plan) as Dictionary
	if request_uses_generated_audio:
		reply_label.text = str(speech_plan.get("reply_text", ""))
	emotion_state = str(speech_plan.get("emotion", "neutral"))
	gesture_weight = clamp(float(speech_plan.get("gesture_intensity", 1.0)), 0.25, 1.8)
	eye_contact = clamp(float(speech_plan.get("eye_contact", 0.7)), 0.0, 1.0)
	var stream: AudioStream
	if request_uses_generated_audio and payload_dict.has("audio_pcm16_base64"):
		var generated:AudioStreamWAV = AudioStreamWAV.new()
		var wav_meta:Dictionary = payload_dict.get("wav", {}) as Dictionary
		generated.format = AudioStreamWAV.FORMAT_16_BITS
		generated.mix_rate = int(wav_meta.get("output_sample_rate", 16000))
		generated.stereo = false
		generated.data = Marshalls.base64_to_raw(str(payload_dict.get("audio_pcm16_base64", "")))
		stream = generated
	else:
		stream = load(TEST_WAV) as AudioStream
	if stream == null:
		status.text = "Motion ready, but WAV playback decode failed"
		return
	last_frame_index = -1
	player.stream = stream; player.play(); playing = true
	var wav_meta:Dictionary = payload_dict.get("wav", {}) as Dictionary
	status.text = "Playing synced audio + %d gesture frames (%.3fs)" % [frames.size(), float(wav_meta.get("duration_s", 0.0))]

func _process(_delta:float)->void:
	if auto_rotate:
		avatar_yaw += _delta * AUTO_ROTATE_SPEED
		$Avatar.rotation.y = avatar_yaw
	_update_camera()
	if not playing or frames.is_empty() or not player.playing:
		return
	var index:int = mini(int(player.get_playback_position()*fps),frames.size()-1)
	var t:float = player.get_playback_position()
	var dt:float = 1.0 / max(fps, 1.0)
	if last_frame_index >= 0:
		dt = max(0.001, float(index - last_frame_index) / max(fps, 1.0))
	last_frame_index = index
	_apply_layered_pose(frames[index], t, dt)
	if index == frames.size()-1: playing = false

func _apply_layered_pose(frame:Dictionary, t:float, dt:float)->void:
	_reset_animated_pose()
	face_energy = lerp(face_energy, _sample_audio_energy(), clamp(dt * 12.0, 0.0, 1.0))
	_apply_gesture_layer(frame)
	_apply_idle_layer(t)
	_apply_head_layer(t)
	_apply_eye_layer(t)
	_apply_finger_layer(t)
	_apply_face_layer(t)

func _reset_animated_pose() -> void:
	for bone:int in animated_bones:
		if bone >= 0:
			skeleton.set_bone_pose_rotation(bone, Quaternion.IDENTITY)

func _apply_gesture_layer(frame:Dictionary)->void:
	var coordinate:Basis = Basis(Vector3(-1,0,0),Vector3(0,1,0),Vector3(0,0,-1))
	for humanoid in frame:
		if not bone_indices.has(humanoid): continue
		var v:Array = frame[humanoid]
		var source:Basis = Basis(Quaternion(float(v[0]),float(v[1]),float(v[2]),float(v[3])))
		var local:Basis = coordinate*source*coordinate.inverse()
		var target:Quaternion = local.get_rotation_quaternion()
		var weight:float = clamp(gesture_weight, 0.0, 1.0)
		if LOWER_BODY_BONES.has(humanoid):
			weight *= 0.08
		elif TORSO_BONES.has(humanoid):
			weight *= 0.82
		skeleton.set_bone_pose_rotation(bone_indices[humanoid], Quaternion.IDENTITY.slerp(target, weight).normalized())

func _apply_idle_layer(t:float)->void:
	var breath:float = sin(t * TAU * 0.23) * 0.018
	_add_local_rotation("spine", Vector3(breath, 0.0, 0.0), 0.45)
	_add_local_rotation("chest", Vector3(breath * 1.4, 0.0, 0.0), 0.55)
	_add_local_rotation("upperChest", Vector3(breath, 0.0, 0.0), 0.45)

func _apply_head_layer(t:float)->void:
	var mood:Dictionary = _emotion_profile()
	var nod:float = sin(t * TAU * 0.37) * 0.018 * float(mood.head_motion)
	var tilt:float = sin(t * TAU * 0.19 + 0.8) * 0.025 * float(mood.head_motion)
	_add_local_rotation("neck", Vector3(nod, 0.0, tilt), 0.65)
	_add_local_rotation("head", Vector3(nod * 0.7, sin(t * TAU * 0.13) * 0.018 * eye_contact, -tilt * 0.4), 0.75)

func _apply_eye_layer(t:float)->void:
	var glance:float = sin(t * TAU * 0.11) * 0.035 * (1.0 - eye_contact)
	for bone in eye_bones:
		if bone >= 0:
			_compose_bone_rotation(bone, Vector3(0.0, glance * 0.35, 0.0))

func _apply_finger_layer(t:float)->void:
	var curl:float = (sin(t * TAU * 0.53) + 1.0) * 0.5 * 0.08 * clamp(gesture_weight, 0.25, 1.8)
	for bone in finger_bones:
		_compose_bone_rotation(bone, Vector3(curl * 0.22, 0.0, 0.0))

func _apply_face_layer(t:float)->void:
	if jaw_bone >= 0:
		var open_amount:float = clamp(face_energy * 0.38, 0.0, 0.24)
		_compose_bone_rotation(jaw_bone, Vector3(open_amount * 0.8, 0.0, 0.0))
	var mood:Dictionary = _emotion_profile()
	_add_local_rotation("leftShoulder", Vector3(0.0, 0.0, float(mood.shoulder_lift)), 0.18)
	_add_local_rotation("rightShoulder", Vector3(0.0, 0.0, -float(mood.shoulder_lift)), 0.18)

func _add_local_rotation(humanoid:String, axis_angles:Vector3, weight:float)->void:
	if not bone_indices.has(humanoid): return
	var index:int = bone_indices[humanoid]
	_compose_bone_rotation(index, axis_angles * clamp(weight, 0.0, 1.0))

func _compose_bone_rotation(index:int, axis_angles:Vector3)->void:
	var current:Quaternion = skeleton.get_bone_pose_rotation(index)
	skeleton.set_bone_pose_rotation(index, (current * Quaternion.from_euler(axis_angles)).normalized())

func _sample_audio_energy()->float:
	var bus:int = AudioServer.get_bus_index("Master")
	if bus < 0: return 0.0
	var left:float = AudioServer.get_bus_peak_volume_left_db(bus, 0)
	var right:float = AudioServer.get_bus_peak_volume_right_db(bus, 0)
	var db:float = max(left, right)
	if db <= -80.0: return 0.0
	return clamp(db_to_linear(db) * 8.0, 0.0, 1.0)

func _emotion_profile()->Dictionary:
	match emotion_state:
		"happy", "excited":
			return {"head_motion":1.35,"shoulder_lift":0.035}
		"sad":
			return {"head_motion":0.55,"shoulder_lift":-0.035}
		"angry":
			return {"head_motion":0.85,"shoulder_lift":0.045}
		"thinking", "curious", "confused":
			return {"head_motion":1.05,"shoulder_lift":0.0}
		"calm", "listening":
			return {"head_motion":0.65,"shoulder_lift":0.0}
		_:
			return {"head_motion":0.85,"shoulder_lift":0.0}
