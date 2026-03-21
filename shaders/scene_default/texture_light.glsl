#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_cam;

out vec3 normal;
out vec2 uv;
out vec3 pos;

void main() {
    mat4 mv = m_cam * m_model; 
    vec4 p = mv * vec4(in_position, 1.0);
	gl_Position = m_proj * p;
    mat3 m_normal = transpose(inverse(mat3(mv)));
    normal = m_normal * in_normal;
    uv = in_texcoord_0;
    pos = p.xyz;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform sampler2D texture0;
uniform vec4 u_light_pos[3];
uniform vec4 u_light_color[3];
uniform float u_specular_strength;
uniform float u_shininess;

in vec3 normal;
in vec3 pos;
in vec2 uv;

void main()
{
    vec3 n = normalize(normal);
    vec3 v = normalize(-pos);
    vec4 tex = texture(texture0, uv);
    vec3 base = tex.rgb;
    vec3 ambient = base * 0.4;
    vec3 lighting = vec3(0.0);
    for (int i = 0; i < 3; i++) {
    vec3 l = normalize(u_light_pos[i].xyz - pos);
        float diff = max(dot(n, l), 0.0);
        vec3 h = normalize(l + v);
        float spec = pow(max(dot(n, h), 0.0), u_shininess) * u_specular_strength;
    lighting += (base * diff + vec3(spec)) * u_light_color[i].rgb;
    }
    vec3 final_color = ambient + lighting;
    fragColor = vec4(final_color, tex.a);
}

#endif
